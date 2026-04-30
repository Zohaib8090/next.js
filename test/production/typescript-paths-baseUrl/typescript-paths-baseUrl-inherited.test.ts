import path from 'path'
import { FileRef, nextTestSetup } from 'e2e-utils'

const describeWebpack =
  process.env.IS_TURBOPACK_TEST === '1' ? describe.skip : describe

// Reproduces https://github.com/vercel/next.js/issues/93336
// Turbopack doesn't support tsconfig outside project root.
describeWebpack('typescript paths with deprecated, inherited baseUrl', () => {
  const fixtureDir = path.join(__dirname, 'fixtures/inherited')
  const { skipped, next } = nextTestSetup({
    files: {
      '../tsconfig.base.json': new FileRef(
        path.join(fixtureDir, 'tsconfig.base.json')
      ),
      '../packages': new FileRef(path.join(fixtureDir, 'packages')),
      'next.config.js': new FileRef(
        path.join(fixtureDir, 'web/next.config.js')
      ),
      'tsconfig.json': new FileRef(path.join(fixtureDir, 'web/tsconfig.json')),
      app: new FileRef(path.join(fixtureDir, 'web/app')),
    },
    subDir: 'web',
    dependencies: {
      // Specifically testing compilerOptions that were deprecated in 6.0.
      // This test would fail in 7.0
      typescript: '^6.0.0',
    },
    // Only next-build is interesting here.
    skipDeployment: true,
  })

  if (skipped) {
    return
  }

  it('should render the page that uses the aliased module', async () => {
    const html = await next.render('/')
    expect(html).toContain('Hello, world!')
  })
})
