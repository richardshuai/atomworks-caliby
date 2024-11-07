<!--
This is a template for pull requests to that will be shown as a checklist to the user.
-->

## 📋 PR Checklist

- [ ] I have ensured that all my commits follow angular commit message conventions.
    > Format: `<type>[optional scope]: <subject>`  
    > Example: `fix(af3): add missing crop transform to the af3 pipeline`
    >
    > This affects semantic versioning as follows:
    > - `fix`: patch version increment (0.0.1 → 0.0.2)
    > - `feat`: minor version increment (0.0.1 → 0.1.0) 
    > - `BREAKING CHANGE`: major version increment (0.0.1 → 1.0.0)
    > - All other types do not affect versioning
    >
    > The format ensures readable changelogs through auto-generation from commit messages.

- [ ] I have run `make format` on the codebase before submitting the PR (this autoformats the code and lints it).

- [ ] I have named the PR in angular PR message format as well (c.f. above), with a sensible tag line that summarizes all the changes in the PR. 
    > This is useful as the name of the PR is the default name of the commit that will be used if you merge with a squash & merge.
    > Format: `<type>[optional scope]: <subject>`  
    > Example: `fix(af3): add missing crop transform to the af3 pipeline`

---

## ℹ️ PR Description

### What changes were made and why?
<!-- Describe the key changes and the reasoning behind them -->
`... your description here ...`

### How were the changes tested?
<!-- Describe how you ensured the changes behaved as expected -->
`... your description here ...`

### Additional Notes
<!-- Any other relevant information -->
`... your description here ...`