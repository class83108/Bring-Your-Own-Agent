"""Eval 測試的 pytest 配置。

提供 --run-eval flag、eval 專用 fixtures、以及 session-scoped 的 EvalStore。
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from tests.eval.framework import compute_prompt_hash
from tests.eval.store import EvalStore

# 預設的 eval 系統提示詞
DEFAULT_EVAL_SYSTEM_PROMPT = """\
# 角色

你是一位專業的程式開發助手，擅長閱讀、理解和修改程式碼。

## 工作流程

每次收到任務時，嚴格遵循以下步驟：

1. **探索**：用 `list_files` 了解專案結構，用 `grep_search` 搜索相關程式碼
2. **閱讀**：用 `read_file` 仔細閱讀相關檔案，理解現有慣例和 pattern
3. **修改**：基於理解進行最小必要的修改
4. **驗證**：用 `bash` 執行 `pytest` 確認修改正確
5. **迭代**：若測試失敗，仔細分析錯誤輸出，找出根因，重複步驟 1-4

**重要**：不要跳過步驟。先探索和閱讀，再修改。修改後一定要跑測試。

## 工具選擇指引

- 不確定檔案位置 → 先用 `list_files` 查看專案結構
- 專案有多個檔案時 → **優先用 `grep_search` 搜索關鍵字**（如函數名、錯誤訊息、\
變數名），快速定位相關檔案，避免逐一 `read_file`
- 搜索函數定義、呼叫位置、特定 pattern → 用 `grep_search`，比逐一 `read_file` 更高效
- 修改前 → 必須先用 `read_file` 理解完整上下文，不要憑記憶修改
- 修改後 → 用 `bash` 跑 `pytest` 驗證，不要假設修改是正確的

**搜索策略**：面對大型專案時，先 `grep_search` 縮小範圍，再 `read_file` 深入理解。\
不要一個一個檔案讀過去。

## 修改原則

- 修改前先閱讀同目錄下的其他檔案，了解既有慣例（命名風格、import 順序、錯誤處理 pattern）
- 新增功能時，沿用專案已有的 pattern，保持風格一致
- 做最小必要的修改，不要過度重構或加入無關的改動

## 開發流程

- 收到「實作功能」需求時：先寫測試，再寫實作，最後跑測試驗證
- 收到「修 bug」需求時：先跑現有測試定位問題，理解根因後再修改
- 測試失敗時：仔細閱讀錯誤訊息，分析原因，針對性修復，不要盲目重試

請使用繁體中文回答。"""


def pytest_addoption(parser: pytest.Parser) -> None:
    """新增 eval 專用命令列參數。"""
    parser.addoption(
        '--run-eval',
        action='store_true',
        default=False,
        help='執行 eval test（會呼叫真實 API）',
    )
    parser.addoption(
        '--eval-agent-version',
        action='store',
        default='unnamed',
        help='Agent 版本標記（如 "v1-baseline"），用於結果追蹤',
    )
    parser.addoption(
        '--eval-system-prompt',
        action='store',
        default=DEFAULT_EVAL_SYSTEM_PROMPT,
        help='eval 使用的系統提示詞（用於 A/B 測試）',
    )
    parser.addoption(
        '--eval-timeout',
        action='store',
        type=float,
        default=300.0,
        help='每個任務的超時時間（秒），預設 300',
    )
    parser.addoption(
        '--eval-model',
        action='store',
        default='claude-sonnet-4-20250514',
        help='eval 使用的模型',
    )
    parser.addoption(
        '--eval-db',
        action='store',
        default='eval-results/eval.db',
        help='eval 結果 SQLite 路徑',
    )
    parser.addoption(
        '--eval-task',
        action='store',
        default=None,
        help='只執行指定的任務（模組名，如 t01_fix_syntax_error）',
    )
    parser.addoption(
        '--eval-notes',
        action='store',
        default=None,
        help='本次 eval run 的備註',
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """根據 --run-eval 決定是否跳過 eval test。"""
    if config.getoption('--run-eval'):
        return

    skip_eval = pytest.mark.skip(reason='需要加 --run-eval 才會執行')
    for item in items:
        if 'eval' in item.keywords:
            item.add_marker(skip_eval)


# --- Session-scoped Fixtures ---


@pytest.fixture(scope='session')
def eval_system_prompt(request: pytest.FixtureRequest) -> str:
    """取得 eval 系統提示詞。"""
    prompt: str = request.config.getoption('--eval-system-prompt')
    return prompt


@pytest.fixture(scope='session')
def eval_timeout(request: pytest.FixtureRequest) -> float:
    """取得 eval 超時設定。"""
    timeout: float = request.config.getoption('--eval-timeout')
    return timeout


@pytest.fixture(scope='session')
def eval_model(request: pytest.FixtureRequest) -> str:
    """取得 eval 模型。"""
    model: str = request.config.getoption('--eval-model')
    return model


@pytest.fixture(scope='session')
def eval_store(request: pytest.FixtureRequest) -> Generator[EvalStore]:
    """建立 session-scoped 的 EvalStore。"""
    db_path: str = request.config.getoption('--eval-db')
    store = EvalStore(db_path)
    yield store
    store.close()


@pytest.fixture(scope='session')
def eval_run_id(
    request: pytest.FixtureRequest,
    eval_store: EvalStore,
    eval_system_prompt: str,
    eval_model: str,
) -> str:
    """建立 session-scoped 的 eval run ID。

    整個 pytest session 的所有任務共用一個 run_id。
    """
    agent_version: str = request.config.getoption('--eval-agent-version')
    notes: str | None = request.config.getoption('--eval-notes')

    run_id = eval_store.create_run(
        agent_version=agent_version,
        system_prompt=eval_system_prompt,
        system_prompt_hash=compute_prompt_hash(eval_system_prompt),
        model=eval_model,
        notes=notes,
    )
    return run_id
