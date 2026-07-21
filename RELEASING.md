# Releasing Chronowire

ChronowireはGit tagとpackage versionを一対一に対応させる。`pyproject.toml`のversionが`X.Y.Z`ならtagは`vX.Y.Z`とする。開発版も同様に、version `0.1.0.dev0`へtag `v0.1.0.dev0`を対応させる。

0.xは設計、実装、性能、API usabilityを検証するdevelopment seriesとし、GitHub tagからのみinstall可能にする。PyPIへはv1.0以降だけを公開する。`release.yml`はmajor version 0のreleaseを機械的に拒否する。

## GitHubからversionを固定してinstallする

tagをGitHubへpushすると、PyPIへ公開する前でも同じrevisionを固定してinstallできる。

```bash
uv add "chronowire @ git+https://github.com/februa/chronowire.git@v0.1.0"
```

```bash
python -m pip install "chronowire @ git+https://github.com/februa/chronowire.git@v0.1.0"
```

## v1.0前のPyPI Trusted Publishing設定

PyPIでprojectまたはpending publisherを作成し、GitHub Trusted Publisherへ次を登録する。

- PyPI project name: `chronowire`
- GitHub owner: `februa`
- Repository: `chronowire`
- Workflow: `release.yml`
- Environment: `pypi`

長期API tokenはrepositoryへ保存しない。GitHub Actionsの`pypi` environmentには、必要に応じてmaintainer approvalを設定する。

## 0.x development tag手順

1. 対象milestoneの検証項目を満たす。
2. `pyproject.toml`と`uv.lock`のversionを同じ0.x versionへ更新する。
3. pytest、Pyright、Ruff、distribution buildを検証する。
4. version変更をcommitする。
5. `v<version>`のannotated tagを作成してpushする。
6. GitHub tagから隔離環境へinstallしてimportとversionを確認する。
7. PyPI publicationを行わない。

## v1.0以降の正式Release手順

1. v1.0のrelease gateを満たす。
2. `pyproject.toml`と`uv.lock`のversionを同じ値へ更新する。
3. pytest、Pyright、Ruff、distribution buildを検証する。
4. version変更をcommitする。
5. `v<version>`のannotated tagを作成してpushする。
6. GitHubで同じtagからReleaseを作成する。
7. `release.yml`のbuildとPyPI publishが成功したことを確認する。
8. PyPIから隔離環境へ`chronowire==<version>`をinstallしてimportを確認する。

現在の`0.1.0`はv0.1 release gateを満たした0.x development releaseであり、PyPIへ公開しない。
