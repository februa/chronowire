# Releasing Chronowire

ChronowireはGit tagとpackage versionを一対一に対応させる。`pyproject.toml`のversionが`X.Y.Z`ならtagは`vX.Y.Z`とする。開発版も同様に、version `0.1.0.dev0`へtag `v0.1.0.dev0`を対応させる。

## GitHubからversionを固定してinstallする

tagをGitHubへpushすると、PyPIへ公開する前でも同じrevisionを固定してinstallできる。

```bash
uv add "chronowire @ git+https://github.com/februa/chronowire.git@v0.1.0.dev0"
```

```bash
python -m pip install "chronowire @ git+https://github.com/februa/chronowire.git@v0.1.0.dev0"
```

## PyPI Trusted Publishingの初回設定

PyPIでprojectまたはpending publisherを作成し、GitHub Trusted Publisherへ次を登録する。

- PyPI project name: `chronowire`
- GitHub owner: `februa`
- Repository: `chronowire`
- Workflow: `release.yml`
- Environment: `pypi`

長期API tokenはrepositoryへ保存しない。GitHub Actionsの`pypi` environmentには、必要に応じてmaintainer approvalを設定する。

## Release手順

1. v0.1またはv0.2のrelease gateを満たす。
2. `pyproject.toml`と`uv.lock`のversionを同じ値へ更新する。
3. pytest、Pyright、Ruff、distribution buildを検証する。
4. version変更をcommitする。
5. `v<version>`のannotated tagを作成してpushする。
6. GitHubで同じtagからReleaseを作成する。
7. `release.yml`のbuildとPyPI publishが成功したことを確認する。
8. PyPIから隔離環境へ`chronowire==<version>`をinstallしてimportを確認する。

現在の`0.1.0.dev0`はv0.1 release gate前の開発版であり、安定版`v0.1.0`としてtag付けしない。
