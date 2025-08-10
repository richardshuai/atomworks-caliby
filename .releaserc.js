module.exports = {
    branches: ['production', 'main'],
    plugins: [
      '@semantic-release/commit-analyzer',
      '@semantic-release/release-notes-generator',
      [
        'semantic-release-pypi',
        {
          pypiPublish: true,
          repoUrl: 'https://test.pypi.org/legacy/',
          distDir: 'dist/',
          setupPy: false,
          pypiToken: process.env.PYPI_TOKEN,
        }
      ],
      '@semantic-release/github',
      [
        '@semantic-release/git',
        {
          assets: ['pyproject.toml'],
          message: 'chore(release): ${nextRelease.version} [skip ci]\n\n${nextRelease.notes}'
        }
      ]
    ]
  };