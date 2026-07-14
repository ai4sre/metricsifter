# Codebase Remediation Design

## Goal

前回のコードベース調査で確認したコアアルゴリズム、実験コード、依存管理、CI、リリース、文書の問題をすべて修正し、再現可能なテストと独立レビューによって回帰を防ぐ。

## Scope

### Core pipeline

1. KDE セグメンテーションは、すべての変化点をちょうど1つのセグメントへ割り当てる。局所最小点と変化点が同じ整数位置でも、隣接セグメントへ重複させない。返却する `labels` と `label_to_values` は同じ所属関係を表す。
2. simple filter は、非欠損値が複数の値へ変化する系列を保持する。`NaN` を挟むことだけを理由に変化を失わない。一方、全欠損、定数、定数と欠損だけの系列、一定傾きの系列を除外する既存方針は維持する。
3. 定数系列を change-point detection へ直接渡した場合、PELT、BinSeg、BottomUp のすべてで例外や偽変化点を返さず、空の変化点リストを返す。
4. 0行の数値系列を直接または `without_simple_filter=True` で渡しても例外を送出せず、空結果を返す。
5. `run_upto_cpd()` は選択列を入力列順で返す。
6. `penalty_adjust` と数値 `bandwidth` は有限な正数だけを受理し、違反時はコンストラクタで説明的な `ValueError` を送出する。`penalty` の文字列は `aic` / `bic` のみを受理する。
7. 列名の重複は内部で誤処理せず、公開入口で説明的な `ValueError` として拒否する。

### Experiment framework

1. `experiments/localization/rcd.py` と `pyrca.py` の未定義名を解消し、RCD wrapper と未知 `building_step` のエラー経路を単体テスト可能にする。
2. 標準テストは、外部Git依存の PyRCA がなくても収集・実行できる。PyRCA固有の統合テストは依存が存在する場合だけ実行し、不在時は明示的にskipする。
3. 実験用依存は、現在のcheckoutを使用し、remote `main` を暗黙に取得しない。実験READMEに実行入口と再現条件を記載する。
4. `requirements-dev.txt` の存在しないincludeを除き、`pyproject.toml` のextrasを正とする導線へ統一する。

### Packaging and dependency management

1. `uv.lock` のプロジェクト版を `pyproject.toml` の `0.2.0` と一致させ、`uv lock --check` で検査できる状態にする。
2. 公開パッケージで未使用の `networkx` をコア依存から外し、必要なテスト・実験側へ限定する。
3. source checkoutだけでなく、buildしたwheelをクリーンな環境へインストールして、importとCLIをsmoke testする。

### CI and release

1. CIを `push` と `pull_request` で実行する。
2. Python 3.10–3.14でコアテストを実行する。外部依存のため3.10–3.11に限定するテストはmarkerで分離する。
3. Ruff、Black、lock整合性、distribution build、wheel smoke testをCIゲートにする。
4. sklearn統合テストは広すぎる `except Exception` で環境破損をskipせず、依存不在だけをskipする。
5. PyPI公開workflowは、同一コミットのCI相当検証とwheel smoke testが成功した成果物だけを公開する。タグとパッケージversionの一致も検査する。

### Documentation and formatting

1. READMEのPython対応範囲、開発セットアップ、テスト、TestPyPI/PyPI公開手順を実際のworkflowと一致させる。
2. RuffとBlackの対象範囲を明確にし、リポジトリ全体を両方に合格させる。
3. 実験READMEの未完の `Run experiment` を、実在する入口と引数を用いた手順へ更新する。

## Architecture and implementation strategy

修正は既存の3段パイプラインを維持し、局所的な入力正規化と不変条件の追加で行う。KDE分割は半開区間として実装し、所属の排他性を保証する。simple filterは非欠損値と差分を分けて評価し、欠損をゼロ差分として扱わない。CPDは検出器構築前に空・定数系列を短絡する。

実験依存とコア依存は分離し、テスト収集時にPyRCAを必須にしない。CIは高速な静的検査、バージョン別コアテスト、成果物検証の順に構成する。公開workflow内でもリリース対象の成果物そのものを検証し、別workflowの実行タイミングへ依存させない。

## Testing strategy

各動作変更はTDDで進める。最初に最小の回帰テストを追加し、現行コードで期待した理由により失敗することを確認する。その後に最小修正を加え、対象テスト、関連モジュール、全体テストの順で確認する。

必須の最終検証は以下とする。

- Python 3.11環境で全pytestを収集・実行できること。
- `ruff check .` が成功すること。
- `black --check .` が成功すること。
- `uv lock --check` が成功すること。
- sdistとwheelのbuildが成功すること。
- buildしたwheelだけをインストールした環境で、`import metricsifter` と `metricsifter --help` が成功すること。

## Review and delivery workflow

作業をコア、実験、依存/CI、文書/リリースのトピックへ分ける。各トピックでは実装担当がTDD、テスト、diff確認によるセルフレビューを行う。続いて別エージェントが仕様適合性を確認し、accept後に別エージェントがコード品質を確認する。指摘がある場合は実装担当へ戻し、同じレビュアーがacceptするまで反復する。

すべてのトピックがacceptされた後、全体レビューと最終検証を行う。1つのfeature branchをpushして単一PRを作成し、GitHub Actionsの全checkが成功するまで原因調査と修正を繰り返す。全check成功後、PRをsquash mergeする。

## Non-goals

- MetricSifterのアルゴリズム自体を別方式へ置き換えない。
- 公開APIを不必要に改名・削除しない。
- 論文実験の数値結果を再生成しない。
- 調査対象外の大規模リファクタリングを行わない。
