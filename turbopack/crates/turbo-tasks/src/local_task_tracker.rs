//! Tracker for the local tasks of a single global task execution.
//!
//! Bundles three pieces of state that must move together:
//!
//! 1. `tasks`: the `LocalTask` slot vector indexed by `LocalTaskId`.
//! 2. `in_flight`: count of local tasks still in `Scheduled` state, plus any in-flight detached
//!    test futures registered via `spawn_detached_for_testing`.
//! 3. `done`: an `Event` notified each time `in_flight` transitions to zero, so the parent's
//!    `wait_for_local_tasks` can wake.
//!
//! All access happens under the surrounding `RwLock<CurrentTaskState>`, so the counter is a
//! plain `u32` and the only synchronization the tracker itself needs is what `Event` provides
//! across the lock boundary (listeners are registered while holding the read guard, then
//! awaited after the guard is dropped).
//!
//! # Invariant
//!
//! Every `in_flight` increment must be matched by exactly one decrement, and the decrement
//! must happen-before any waiter that started after the increment can return. The two
//! production call sites
//! - [`create`] (increment) paired with [`complete`] (decrement) for `Scheduled → Done`
//!   transitions, and the test-only path
//! - [`register_detached`] (increment) paired with [`complete_detached`] (decrement)
//!
//! both maintain this. Late registrations are only allowed from inside an already-counted
//! task body, so by the time the parent calls `wait_for_local_tasks` the counter reflects all
//! pending work.
//!
//! [`create`]: LocalTaskTracker::create
//! [`complete`]: LocalTaskTracker::complete
//! [`register_detached`]: LocalTaskTracker::register_detached
//! [`complete_detached`]: LocalTaskTracker::complete_detached

use crate::{
    OutputContent,
    event::{Event, EventListener},
    id::LocalTaskId,
    task::local_task::LocalTask,
};

pub(crate) struct LocalTaskTracker {
    /// Slot vector for `LocalTask` entries. One-indexed via `LocalTaskId`.
    tasks: Vec<LocalTask>,
    /// Count of `tasks` entries still in `Scheduled` state, plus any in-flight detached test
    /// futures. Decrementing to zero notifies `done`.
    in_flight: u32,
    /// Notified each time `in_flight` transitions to zero.
    done: Event,
}

impl LocalTaskTracker {
    pub(crate) fn new() -> Self {
        Self {
            tasks: Vec::new(),
            in_flight: 0,
            done: Event::new(|| || "LocalTaskTracker::done".to_string()),
        }
    }

    pub(crate) fn get(&self, id: LocalTaskId) -> &LocalTask {
        // local task ids are one-indexed (they use NonZeroU32)
        &self.tasks[(*id as usize) - 1]
    }

    /// Push a new `Scheduled` local task into the slot vector and increment the in-flight
    /// counter. Returns the new task's id.
    ///
    /// The increment is balanced by [`complete`] when the task transitions to `Done`. The
    /// codebase relies on tasks always reaching `complete` (panics in the body abort the
    /// process upstream); we intentionally do not provide RAII unwind safety here.
    ///
    /// [`complete`]: LocalTaskTracker::complete
    pub(crate) fn create(&mut self, task: LocalTask) -> LocalTaskId {
        debug_assert!(
            matches!(task, LocalTask::Scheduled { .. }),
            "newly created local tasks must start in the Scheduled state"
        );
        self.tasks.push(task);
        self.in_flight += 1;
        // generate a one-indexed id from len() -- we just pushed so len() is >= 1
        if cfg!(debug_assertions) {
            LocalTaskId::try_from(u32::try_from(self.tasks.len()).unwrap()).unwrap()
        } else {
            // SAFETY: len() is >= 1 because we just pushed.
            unsafe { LocalTaskId::new_unchecked(self.tasks.len() as u32) }
        }
    }

    /// Transition the slot for `id` from `Scheduled` to `Done`, decrement the in-flight
    /// counter, and notify the collective `done` event if it reached zero.
    ///
    /// Returns the per-task `done_event` extracted from the previous `Scheduled` variant so
    /// the caller can notify per-task consumers (`try_read_local_output` waiters) outside the
    /// surrounding lock.
    pub(crate) fn complete(&mut self, id: LocalTaskId, output: OutputContent) -> Event {
        let slot = &mut self.tasks[(*id as usize) - 1];
        let prev = std::mem::replace(slot, LocalTask::Done { output });
        let LocalTask::Scheduled { done_event } = prev else {
            panic!("local task finished, but was not in the scheduled state?");
        };
        self.dec_in_flight();
        done_event
    }

    /// Test-only: register an in-flight detached future (`spawn_detached_for_testing`). No
    /// `LocalTask` slot is allocated; this just bumps the counter. Balanced by a matching
    /// [`dec_in_flight`] when the wrapped future completes.
    ///
    /// [`dec_in_flight`]: LocalTaskTracker::dec_in_flight
    pub(crate) fn register_detached(&mut self) {
        self.in_flight += 1;
    }

    /// Decrement the in-flight counter and notify the collective `done` event if it reached
    /// zero. Used by the test-only detached path; the production path goes through
    /// [`complete`] which decrements as part of the slot transition.
    ///
    /// [`complete`]: LocalTaskTracker::complete
    pub(crate) fn dec_in_flight(&mut self) {
        debug_assert!(
            self.in_flight > 0,
            "LocalTaskTracker::dec_in_flight without matching increment"
        );
        self.in_flight -= 1;
        if self.in_flight == 0 {
            self.done.notify(usize::MAX);
        }
    }

    /// Current in-flight count. Cheap snapshot for the early-return path in
    /// `wait_for_local_tasks`.
    pub(crate) fn in_flight(&self) -> u32 {
        self.in_flight
    }

    /// Listen for the next "in-flight reached zero" notification. Used by
    /// `wait_for_local_tasks` together with `in_flight()` for the standard double-check
    /// pattern that avoids lost wakeups.
    pub(crate) fn listen(&self) -> EventListener {
        self.done.listen()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_output() -> OutputContent {
        OutputContent::Link(crate::raw_vc::RawVc::TaskOutput(
            crate::TaskId::try_from(1).unwrap(),
        ))
    }

    /// `create` increments `in_flight`; `complete` performs the slot transition AND
    /// decrements in one step.
    #[test]
    fn create_then_complete_balances_in_flight() {
        let mut tracker = LocalTaskTracker::new();
        assert_eq!(tracker.in_flight(), 0);

        let id = tracker.create(LocalTask::Scheduled {
            done_event: Event::new(|| || "test".to_string()),
        });
        assert_eq!(tracker.in_flight(), 1);

        tracker.complete(id, dummy_output());
        assert_eq!(tracker.in_flight(), 0);
    }

    #[test]
    fn detached_inc_dec_balances_in_flight() {
        let mut tracker = LocalTaskTracker::new();
        tracker.register_detached();
        tracker.register_detached();
        assert_eq!(tracker.in_flight(), 2);
        tracker.dec_in_flight();
        assert_eq!(tracker.in_flight(), 1);
        tracker.dec_in_flight();
        assert_eq!(tracker.in_flight(), 0);
    }

    #[test]
    fn complete_returns_done_event_from_scheduled_slot() {
        let mut tracker = LocalTaskTracker::new();
        let id = tracker.create(LocalTask::Scheduled {
            done_event: Event::new(|| || "per-task".to_string()),
        });
        // Sanity: just confirm the returned Event can be notified without panic.
        let done_event = tracker.complete(id, dummy_output());
        done_event.notify(usize::MAX);
    }
}
