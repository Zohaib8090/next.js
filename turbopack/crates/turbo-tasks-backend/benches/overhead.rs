use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, black_box};
use futures::{FutureExt, StreamExt, stream::FuturesUnordered};
use tokio::spawn;
use turbo_tasks::{TurboTasks, unmark_top_level_task_may_leak_eventually_consistent_state};
use turbo_tasks_backend::{BackendOptions, TurboTasksBackend, noop_backing_storage};

#[global_allocator]
static ALLOC: turbo_tasks_malloc::TurboMalloc = turbo_tasks_malloc::TurboMalloc;

// Tunable task: busy-wait for a given duration
#[inline(never)]
fn busy_task(duration: Duration) {
    let start = Instant::now();
    while start.elapsed() < duration {
        std::hint::spin_loop();
    }
}

// Simulate running the task inside turbo-tasks (replace with actual turbo-tasks API)
#[turbo_tasks::function]
fn busy_turbo(key: u64, duration: Duration) {
    busy_task(black_box(duration));
    black_box(key); // consume the key, we need it to be part of the cache key.
}

// Zero-work tasks for isolating dispatch / future-construction overhead from user work.
// These are sync-bodied so they exercise the `FunctionMode` impls (no internal awaits).
//
// Arity-0 has no cache key, so it can only be benchmarked in cache-hit mode (after the first
// call, every subsequent call hits the same cached entry).
#[turbo_tasks::function]
fn noop_arity0() {}

#[turbo_tasks::function]
fn noop_arity1(key: u64) {
    black_box(key);
}

#[turbo_tasks::function]
fn noop_arity2(key: u64, extra: u64) {
    black_box(key);
    black_box(extra);
}

// Async variants. The internal `yield_now().await` forces a real suspension point. This is the
// shape that exposes the suspect difference between canary and HEAD: on HEAD the user-arg `&T`
// references must live across the suspension in the future state (alongside the Arc keepalive),
// whereas on canary args were cloned to owned values before the future was constructed.
#[turbo_tasks::function]
async fn noop_arity0_async() {
    tokio::task::yield_now().await;
}

#[turbo_tasks::function]
async fn noop_arity1_async(key: u64) {
    tokio::task::yield_now().await;
    black_box(key);
}

#[turbo_tasks::function]
async fn noop_arity2_async(key: u64, extra: u64) {
    tokio::task::yield_now().await;
    black_box(key);
    black_box(extra);
}

pub fn overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("task_overhead");
    group.sample_size(100);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .disable_lifo_slot()
        .worker_threads(1)
        .thread_name("tokio-thread")
        .enable_all()
        .build()
        .unwrap();

    let rt_parallel = tokio::runtime::Builder::new_multi_thread()
        .disable_lifo_slot()
        .thread_name("tokio-parallel-thread")
        .enable_all()
        .build()
        .unwrap();

    // Test durations between 10us and 1ms.  This enables two things
    // 1. ensure that our busy task is working correctly, we should see uncached times scale with
    //    this metric
    // 2. see if there are effects related to how long await points take
    for micros in [1, 10, 100, 1000] {
        let duration = Duration::from_micros(micros);

        group.bench_with_input(BenchmarkId::new("direct", micros), &duration, |b, &d| {
            b.iter(|| busy_task(black_box(d)))
        });

        group.bench_with_input(BenchmarkId::new("tokio", micros), &duration, |b, &d| {
            b.to_async(&rt).iter_custom(move |iters| {
                spawn(async move {
                    let start = Instant::now();
                    for _ in 0..iters {
                        spawn(async move {
                            busy_task(black_box(d));
                        })
                        .await
                        .unwrap();
                    }
                    start.elapsed()
                })
                .then(|r| async { r.unwrap() })
            });
        });

        group.bench_with_input(
            BenchmarkId::new("turbo-uncached", micros),
            &duration,
            |b, &d| {
                run_turbo::<Uncached>(&rt, b, d, false);
            },
        );

        group.bench_with_input(
            BenchmarkId::new("turbo-cached-same-keys", micros),
            &duration,
            |b, &d| {
                run_turbo::<CachedSame>(&rt, b, d, false);
            },
        );

        group.bench_with_input(
            BenchmarkId::new("turbo-cached-different-keys", micros),
            &duration,
            |b, &d| {
                run_turbo::<CachedDifferent>(&rt, b, d, false);
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tokio-parallel", micros),
            &duration,
            |b, &d| {
                b.to_async(&rt_parallel).iter_custom(move |iters| {
                    spawn(async move {
                        let start = Instant::now();
                        let mut futures = (0..iters)
                            .map(|_| {
                                spawn(async move {
                                    busy_task(black_box(d));
                                })
                            })
                            .collect::<FuturesUnordered<_>>();
                        while futures.next().await.is_some() {}
                        start.elapsed()
                    })
                    .then(|r| async { r.unwrap() })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("turbo-uncached-parallel", micros),
            &duration,
            |b, &d| {
                run_turbo::<Uncached>(&rt_parallel, b, d, true);
            },
        );
    }
    group.finish();

    // Isolate dispatch / future-construction overhead from user work using zero-work tasks.
    // Compares arity-0, arity-1, and arity-2 (sync & async) to surface costs that scale with
    // arg count (per-arg downcast, future-state size, self-referential capture in arity ≥ 1)
    // and costs that scale with internal await suspensions (the async variants).
    //
    // Sample size is high (500) because the cached-path measurements are sub-microsecond and
    // we are looking for ~2% effects.
    let mut dispatch_group = c.benchmark_group("task_dispatch_overhead");
    dispatch_group.sample_size(500);

    // Sync arity-1 / arity-2 across all three modes — these can vary by key.
    dispatch_group.bench_function("arity1-uncached", |b| {
        run_turbo_noop_arity1::<Uncached>(&rt, b);
    });
    dispatch_group.bench_function("arity1-cached-same-keys", |b| {
        run_turbo_noop_arity1::<CachedSame>(&rt, b);
    });
    dispatch_group.bench_function("arity1-cached-different-keys", |b| {
        run_turbo_noop_arity1::<CachedDifferent>(&rt, b);
    });

    dispatch_group.bench_function("arity2-uncached", |b| {
        run_turbo_noop_arity2::<Uncached>(&rt, b);
    });
    dispatch_group.bench_function("arity2-cached-same-keys", |b| {
        run_turbo_noop_arity2::<CachedSame>(&rt, b);
    });
    dispatch_group.bench_function("arity2-cached-different-keys", |b| {
        run_turbo_noop_arity2::<CachedDifferent>(&rt, b);
    });

    // Sync arity-0 (no key, only cache-hit path).
    dispatch_group.bench_function("arity0-cached", |b| {
        run_turbo_noop_arity0(&rt, b);
    });

    // Async variants — internal yield_now().await forces a real suspension, exposing any
    // future-state-size cost from holding the Arc keepalive + arg refs across the await.
    dispatch_group.bench_function("arity0-async-cached", |b| {
        run_turbo_noop_arity0_async(&rt, b);
    });
    dispatch_group.bench_function("arity1-async-uncached", |b| {
        run_turbo_noop_arity1_async::<Uncached>(&rt, b);
    });
    dispatch_group.bench_function("arity1-async-cached-same-keys", |b| {
        run_turbo_noop_arity1_async::<CachedSame>(&rt, b);
    });
    dispatch_group.bench_function("arity1-async-cached-different-keys", |b| {
        run_turbo_noop_arity1_async::<CachedDifferent>(&rt, b);
    });
    dispatch_group.bench_function("arity2-async-uncached", |b| {
        run_turbo_noop_arity2_async::<Uncached>(&rt, b);
    });
    dispatch_group.bench_function("arity2-async-cached-same-keys", |b| {
        run_turbo_noop_arity2_async::<CachedSame>(&rt, b);
    });
    dispatch_group.bench_function("arity2-async-cached-different-keys", |b| {
        run_turbo_noop_arity2_async::<CachedDifferent>(&rt, b);
    });

    dispatch_group.finish();
}

trait TurboMode {
    fn key(index: u64) -> u64;
    fn is_cached() -> bool;
}
struct Uncached;
impl TurboMode for Uncached {
    fn key(index: u64) -> u64 {
        index
    }

    fn is_cached() -> bool {
        false
    }
}
struct CachedSame;
impl TurboMode for CachedSame {
    fn key(_index: u64) -> u64 {
        0
    }

    fn is_cached() -> bool {
        true
    }
}
struct CachedDifferent;
impl TurboMode for CachedDifferent {
    fn key(index: u64) -> u64 {
        index
    }

    fn is_cached() -> bool {
        true
    }
}

fn run_turbo_noop_arity0(rt: &tokio::runtime::Runtime, b: &mut criterion::Bencher<'_>) {
    b.to_async(rt).iter_custom(|iters| {
        let tt = TurboTasks::new(TurboTasksBackend::new(
            BackendOptions {
                storage_mode: None,
                ..Default::default()
            },
            noop_backing_storage(),
        ));

        async move {
            tt.run(async move {
                unmark_top_level_task_may_leak_eventually_consistent_state();
                // Warm the cache once; arity-0 has no key, so all calls hit the same entry.
                black_box(noop_arity0().await?);
                let start = Instant::now();
                for _ in 0..iters {
                    black_box(noop_arity0().await?);
                }
                Ok(start.elapsed())
            })
            .await
            .unwrap()
        }
    });
}

fn run_turbo_noop_arity1<Mode: TurboMode>(
    rt: &tokio::runtime::Runtime,
    b: &mut criterion::Bencher<'_>,
) {
    b.to_async(rt).iter_custom(|iters| {
        let tt = TurboTasks::new(TurboTasksBackend::new(
            BackendOptions {
                storage_mode: None,
                ..Default::default()
            },
            noop_backing_storage(),
        ));

        async move {
            tt.run(async move {
                unmark_top_level_task_may_leak_eventually_consistent_state();
                if Mode::is_cached() {
                    for i in 0..iters {
                        black_box(noop_arity1(i).await?);
                    }
                }
                let start = Instant::now();
                for i in 0..iters {
                    black_box(noop_arity1(Mode::key(i)).await?);
                }
                Ok(start.elapsed())
            })
            .await
            .unwrap()
        }
    });
}

fn run_turbo_noop_arity2<Mode: TurboMode>(
    rt: &tokio::runtime::Runtime,
    b: &mut criterion::Bencher<'_>,
) {
    b.to_async(rt).iter_custom(|iters| {
        let tt = TurboTasks::new(TurboTasksBackend::new(
            BackendOptions {
                storage_mode: None,
                ..Default::default()
            },
            noop_backing_storage(),
        ));

        async move {
            tt.run(async move {
                unmark_top_level_task_may_leak_eventually_consistent_state();
                if Mode::is_cached() {
                    for i in 0..iters {
                        black_box(noop_arity2(i, 0).await?);
                    }
                }
                let start = Instant::now();
                for i in 0..iters {
                    black_box(noop_arity2(Mode::key(i), 0).await?);
                }
                Ok(start.elapsed())
            })
            .await
            .unwrap()
        }
    });
}

fn run_turbo_noop_arity0_async(rt: &tokio::runtime::Runtime, b: &mut criterion::Bencher<'_>) {
    b.to_async(rt).iter_custom(|iters| {
        let tt = TurboTasks::new(TurboTasksBackend::new(
            BackendOptions {
                storage_mode: None,
                ..Default::default()
            },
            noop_backing_storage(),
        ));

        async move {
            tt.run(async move {
                unmark_top_level_task_may_leak_eventually_consistent_state();
                black_box(noop_arity0_async().await?);
                let start = Instant::now();
                for _ in 0..iters {
                    black_box(noop_arity0_async().await?);
                }
                Ok(start.elapsed())
            })
            .await
            .unwrap()
        }
    });
}

fn run_turbo_noop_arity1_async<Mode: TurboMode>(
    rt: &tokio::runtime::Runtime,
    b: &mut criterion::Bencher<'_>,
) {
    b.to_async(rt).iter_custom(|iters| {
        let tt = TurboTasks::new(TurboTasksBackend::new(
            BackendOptions {
                storage_mode: None,
                ..Default::default()
            },
            noop_backing_storage(),
        ));

        async move {
            tt.run(async move {
                unmark_top_level_task_may_leak_eventually_consistent_state();
                if Mode::is_cached() {
                    for i in 0..iters {
                        black_box(noop_arity1_async(i).await?);
                    }
                }
                let start = Instant::now();
                for i in 0..iters {
                    black_box(noop_arity1_async(Mode::key(i)).await?);
                }
                Ok(start.elapsed())
            })
            .await
            .unwrap()
        }
    });
}

fn run_turbo_noop_arity2_async<Mode: TurboMode>(
    rt: &tokio::runtime::Runtime,
    b: &mut criterion::Bencher<'_>,
) {
    b.to_async(rt).iter_custom(|iters| {
        let tt = TurboTasks::new(TurboTasksBackend::new(
            BackendOptions {
                storage_mode: None,
                ..Default::default()
            },
            noop_backing_storage(),
        ));

        async move {
            tt.run(async move {
                unmark_top_level_task_may_leak_eventually_consistent_state();
                if Mode::is_cached() {
                    for i in 0..iters {
                        black_box(noop_arity2_async(i, 0).await?);
                    }
                }
                let start = Instant::now();
                for i in 0..iters {
                    black_box(noop_arity2_async(Mode::key(i), 0).await?);
                }
                Ok(start.elapsed())
            })
            .await
            .unwrap()
        }
    });
}

fn run_turbo<Mode: TurboMode>(
    rt: &tokio::runtime::Runtime,
    b: &mut criterion::Bencher<'_>,
    d: Duration,
    is_parallel: bool,
) {
    b.to_async(rt).iter_custom(|iters| {
        // It is important to create the tt instance here to ensure the cache is not shared across
        // iterations.
        let tt = TurboTasks::new(TurboTasksBackend::new(
            BackendOptions {
                storage_mode: None,
                ..Default::default()
            },
            noop_backing_storage(),
        ));

        async move {
            tt.run(async move {
                unmark_top_level_task_may_leak_eventually_consistent_state();
                // If cached run once outside the loop to ensure the tasks are cached.
                if Mode::is_cached() {
                    for i in 0..iters {
                        // Precache all possible tasks even if we might only check a few below.
                        // This ensures we are testing a large cache
                        // Do not use Mode::key here, to create a large task set
                        black_box(busy_turbo(i, black_box(d)).await?);
                    }
                }
                if is_parallel {
                    let mut vcs = Vec::with_capacity(iters as usize);
                    let start = Instant::now();
                    vcs.extend(
                        (0..iters).map(|i| black_box(busy_turbo(Mode::key(i), black_box(d)))),
                    );
                    for vc in vcs {
                        vc.await?;
                    }
                    Ok(start.elapsed())
                } else {
                    let start = Instant::now();
                    for i in 0..iters {
                        black_box(busy_turbo(Mode::key(i), black_box(d)).await?);
                    }
                    Ok(start.elapsed())
                }
            })
            .await
            .unwrap()
        }
    });
}
