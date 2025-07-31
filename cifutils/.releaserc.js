module.exports = {
    branches: ['production', 'docs_setup'],
    plugins: [
      '@semantic-release/commit-analyzer',
      '@semantic-release/release-notes-generator',
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
  