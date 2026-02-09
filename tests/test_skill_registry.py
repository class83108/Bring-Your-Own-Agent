"""Skill Registry 測試模組。

根據 docs/features/skill.feature 規格撰寫測試案例。
涵蓋：
- Rule: Skill 應支援註冊與管理
- Rule: Skill 應支援兩階段載入
- Rule: Skill 應能擴充 System Prompt
- Rule: Skill 應能列出與查詢
"""

from __future__ import annotations

import allure
import pytest

from agent_core.skills import Skill, SkillRegistry

# =============================================================================
# Rule: Skill 應支援註冊與管理
# =============================================================================


@allure.feature('Skill 技能系統')
@allure.story('Skill 應支援註冊與管理')
class TestSkillRegistration:
    """Skill 註冊功能測試。"""

    @allure.title('註冊一個 Skill')
    def test_register_skill(self) -> None:
        """Scenario: 註冊一個 Skill。"""
        registry = SkillRegistry()
        skill = Skill(
            name='fitness',
            description='健身紀錄助手',
            instructions='你擅長健身建議，幫助使用者記錄訓練。',
        )

        registry.register(skill)

        assert 'fitness' in registry.list_skills()

    @allure.title('註冊多個 Skill')
    def test_register_multiple_skills(self) -> None:
        """Scenario: 註冊多個 Skill。"""
        registry = SkillRegistry()
        registry.register(
            Skill(
                name='fitness',
                description='健身助手',
                instructions='你擅長健身',
            )
        )
        registry.register(
            Skill(
                name='nutrition',
                description='營養助手',
                instructions='你擅長營養',
            )
        )

        assert len(registry.list_skills()) == 2

    @allure.title('不允許重複註冊同名 Skill')
    def test_duplicate_skill_name_raises_error(self) -> None:
        """Scenario: 不允許重複註冊同名 Skill。"""
        registry = SkillRegistry()
        registry.register(
            Skill(
                name='fitness',
                description='健身助手',
                instructions='你擅長健身',
            )
        )

        with pytest.raises(ValueError, match='fitness'):
            registry.register(
                Skill(
                    name='fitness',
                    description='另一個健身助手',
                    instructions='不同的內容',
                )
            )


# =============================================================================
# Rule: Skill 應支援兩階段載入
# =============================================================================


@allure.feature('Skill 技能系統')
@allure.story('Skill 應支援兩階段載入')
class TestSkillTwoPhaseLoading:
    """Skill 兩階段載入測試。"""

    @allure.title('Phase 1 — 只載入 Skill 描述清單')
    def test_get_skill_descriptions(self) -> None:
        """Scenario: Phase 1 — 只載入 Skill 描述清單。"""
        registry = SkillRegistry()
        registry.register(
            Skill(
                name='fitness',
                description='健身紀錄助手',
                instructions='完整健身指令（不應出現在描述中）',
            )
        )
        registry.register(
            Skill(
                name='nutrition',
                description='營養建議',
                instructions='完整營養指令（不應出現在描述中）',
            )
        )

        descriptions = registry.get_skill_descriptions()

        assert 'fitness' in descriptions
        assert '健身紀錄助手' in descriptions
        assert 'nutrition' in descriptions
        assert '營養建議' in descriptions
        # Phase 1 不應包含完整 instructions
        assert '完整健身指令' not in descriptions
        assert '完整營養指令' not in descriptions

    @allure.title('Phase 1 — 隱藏 disable_model_invocation 的 Skill')
    def test_disable_model_invocation_hides_description(self) -> None:
        """Scenario: Phase 1 — 隱藏 disable_model_invocation 的 Skill。"""
        registry = SkillRegistry()
        registry.register(
            Skill(
                name='visible',
                description='可見的 Skill',
                instructions='可見指令',
            )
        )
        registry.register(
            Skill(
                name='hidden',
                description='隱藏的 Skill',
                instructions='隱藏指令',
                disable_model_invocation=True,
            )
        )

        descriptions = registry.get_skill_descriptions()

        assert 'visible' in descriptions
        assert '可見的 Skill' in descriptions
        assert 'hidden' not in descriptions
        assert '隱藏的 Skill' not in descriptions

    @allure.title('Phase 2 — 啟用 Skill 載入完整指令')
    def test_activate_skill(self) -> None:
        """Scenario: Phase 2 — 啟用 Skill 載入完整指令。"""
        registry = SkillRegistry()
        registry.register(
            Skill(
                name='fitness',
                description='健身助手',
                instructions='你擅長健身建議',
            )
        )

        registry.activate('fitness')

        assert 'fitness' in registry.list_active_skills()

    @allure.title('停用已啟用的 Skill')
    def test_deactivate_skill(self) -> None:
        """Scenario: 停用已啟用的 Skill。"""
        registry = SkillRegistry()
        registry.register(
            Skill(
                name='fitness',
                description='健身助手',
                instructions='你擅長健身建議',
            )
        )
        registry.activate('fitness')

        registry.deactivate('fitness')

        assert 'fitness' not in registry.list_active_skills()

    @allure.title('啟用不存在的 Skill 應拋出 KeyError')
    def test_activate_nonexistent_skill_raises_error(self) -> None:
        """啟用不存在的 Skill 應拋出 KeyError。"""
        registry = SkillRegistry()

        with pytest.raises(KeyError, match='unknown'):
            registry.activate('unknown')

    @allure.title('重複啟用同一個 Skill 不應報錯')
    def test_activate_already_active_is_idempotent(self) -> None:
        """重複啟用同一個 Skill 不應報錯。"""
        registry = SkillRegistry()
        registry.register(
            Skill(
                name='fitness',
                description='健身助手',
                instructions='健身指令',
            )
        )

        registry.activate('fitness')
        registry.activate('fitness')

        assert registry.list_active_skills().count('fitness') == 1


# =============================================================================
# Rule: Skill 應能擴充 System Prompt
# =============================================================================


@allure.feature('Skill 技能系統')
@allure.story('Skill 應能擴充 System Prompt')
class TestSkillSystemPrompt:
    """Skill 擴充 System Prompt 測試。"""

    @allure.title('合併基礎提示、描述清單與已啟用 Skill 指令')
    def test_combined_prompt_with_active_skill(self) -> None:
        """Scenario: 合併基礎提示、描述清單與已啟用 Skill 指令。"""
        registry = SkillRegistry()
        registry.register(
            Skill(
                name='fitness',
                description='健身助手',
                instructions='你擅長健身建議',
            )
        )
        registry.register(
            Skill(
                name='nutrition',
                description='營養助手',
                instructions='你擅長營養建議',
            )
        )

        # 只啟用 fitness
        registry.activate('fitness')

        result = registry.get_combined_system_prompt('你是一位助手')

        # 應包含基礎 prompt
        assert '你是一位助手' in result
        # 應包含 Skill 描述清單（Phase 1）
        assert '健身助手' in result
        assert '營養助手' in result
        # 應包含已啟用 Skill 的完整 instructions（Phase 2）
        assert '你擅長健身建議' in result
        # 不應包含未啟用 Skill 的完整 instructions
        assert '你擅長營養建議' not in result

    @allure.title('無 Skill 時只回傳基礎提示')
    def test_no_skills_returns_base_prompt(self) -> None:
        """Scenario: 無 Skill 時只回傳基礎提示。"""
        registry = SkillRegistry()

        result = registry.get_combined_system_prompt('你是一位助手')

        assert result == '你是一位助手'

    @allure.title('有 Skill 但都未啟用時，只帶描述清單')
    def test_skills_registered_but_none_active(self) -> None:
        """有 Skill 但都未啟用時，只帶描述清單。"""
        registry = SkillRegistry()
        registry.register(
            Skill(
                name='fitness',
                description='健身助手',
                instructions='你擅長健身建議',
            )
        )

        result = registry.get_combined_system_prompt('你是一位助手')

        assert '你是一位助手' in result
        # 應有描述清單
        assert '健身助手' in result
        # 不應有完整 instructions
        assert '你擅長健身建議' not in result


# =============================================================================
# Rule: Skill 應能列出與查詢
# =============================================================================


@allure.feature('Skill 技能系統')
@allure.story('Skill 應能列出與查詢')
class TestSkillQuery:
    """Skill 列出與查詢測試。"""

    @allure.title('列出所有已註冊的 Skill')
    def test_list_skills(self) -> None:
        """Scenario: 列出所有已註冊的 Skill。"""
        registry = SkillRegistry()
        registry.register(
            Skill(
                name='fitness',
                description='健身助手',
                instructions='健身指令',
            )
        )
        registry.register(
            Skill(
                name='nutrition',
                description='營養助手',
                instructions='營養指令',
            )
        )

        names = registry.list_skills()

        assert 'fitness' in names
        assert 'nutrition' in names

    @allure.title('依名稱取得 Skill')
    def test_get_skill_by_name(self) -> None:
        """Scenario: 依名稱取得 Skill。"""
        registry = SkillRegistry()
        skill = Skill(
            name='fitness',
            description='健身助手',
            instructions='你擅長健身建議',
        )
        registry.register(skill)

        found = registry.get('fitness')

        assert found is not None
        assert found.name == 'fitness'
        assert found.description == '健身助手'
        assert found.instructions == '你擅長健身建議'

    @allure.title('查詢不存在的 Skill')
    def test_get_nonexistent_skill_returns_none(self) -> None:
        """Scenario: 查詢不存在的 Skill。"""
        registry = SkillRegistry()

        result = registry.get('unknown')

        assert result is None
