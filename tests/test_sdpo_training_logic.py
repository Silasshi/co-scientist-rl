import pytest

from tinker_cookbook.recipes.rl_co_scientist.SDPO.SDPO import (
    Config,
    build_research_plan_prompt,
    build_self_teacher_prompt,
    check_format_compliance,
    compute_non_solution_penalty,
    compute_rubric_reward_from_loose_levels,
    compute_final_reward,
    compute_group_advantages,
    compute_think_penalty,
    compute_sdpo_success_cutoff,
    extract_non_solution_text,
    extract_think_text,
    extract_solution_text,
    find_solution_content_span,
    get_trainability_drop_reason,
    is_template_copy_solution,
    is_sample_trainable,
    normalize_policy_output_for_grader,
    should_drop_non_solution_too_long,
    should_drop_think_too_long,
)


def test_prompt_solution_only_contract():
    prompt = build_research_plan_prompt(
        scenario="Test scenario",
        policy_output_mode="solution_only",
    )
    assert "<solution>" in prompt
    assert "</solution>" in prompt
    assert "<think>" not in prompt
    assert "</think>" not in prompt
    assert "A complete and self-contained research plan in at most 750 words." not in prompt


def test_prompt_think_solution_contract():
    prompt = build_research_plan_prompt(
        scenario="Test scenario",
        policy_output_mode="think_solution",
    )
    assert "<think>" in prompt
    assert "</think>" in prompt
    assert "<solution>" in prompt
    assert "</solution>" in prompt
    assert "A complete and self-contained research plan in at most 750 words." not in prompt


def test_prompt_invalid_mode_raises():
    with pytest.raises(ValueError, match="Invalid policy_output_mode"):
        build_research_plan_prompt(
            scenario="Test scenario",
            policy_output_mode="bad_mode",
        )


def test_self_teacher_prompt_solution_only_instruction():
    prompt = build_self_teacher_prompt(
        scenario="Question",
        grader_feedback="Feedback",
        successful_previous_rollout=None,
        policy_output_mode="solution_only",
    )
    assert "ONLY the <solution> tags" in prompt
    assert "Keep your <think> process brief." not in prompt


def test_self_teacher_prompt_think_solution_instruction():
    prompt = build_self_teacher_prompt(
        scenario="Question",
        grader_feedback="Feedback",
        successful_previous_rollout=None,
        policy_output_mode="think_solution",
    )
    assert "Remember to use the <think> and <solution> tags." in prompt
    assert "Keep your <think> process brief." in prompt


def test_extract_solution_text_requires_tags():
    text = "<solution>Hello world</solution>"
    assert extract_solution_text(text, require_tags=True) == "Hello world"
    assert extract_solution_text("No tags here", require_tags=True) is None


def test_extract_solution_ignores_solution_tags_inside_think():
    text = (
        "<think>do not use <solution>fake plan</solution> in reasoning</think>\n"
        "<solution>real plan</solution>"
    )
    assert extract_solution_text(text, require_tags=True) == "real plan"


def test_extract_solution_missing_when_only_inside_think():
    text = "<think>quoted <solution>fake plan</solution> only</think>"
    assert extract_solution_text(text, require_tags=True) is None
    assert not check_format_compliance(text, max_words=750)


def test_extract_think_text_basic_and_requires_tags():
    text = "<think>brief reasoning</think><solution>final plan</solution>"
    assert extract_think_text(text, require_tags=True) == "brief reasoning"
    assert extract_think_text("<solution>final plan</solution>", require_tags=True) is None


def test_extract_think_ignores_embedded_think_inside_solution():
    text = (
        "<think>real chain</think>"
        "<solution>quoted <think>not real chain</think> in content</solution>"
    )
    assert extract_think_text(text, require_tags=True) == "real chain"


def test_extract_non_solution_text_removes_solution_content():
    text = (
        "<think>brief reason</think>"
        "<solution>this should be removed</solution>"
        "tail note"
    )
    non_solution = extract_non_solution_text(text)
    assert "brief" in non_solution
    assert "tail" in non_solution
    assert "removed" not in non_solution


def test_is_template_copy_solution_detects_known_placeholder():
    assert is_template_copy_solution(
        "A complete and self-contained research plan in at most 750 words."
    )
    assert is_template_copy_solution(
        "A complete and self-contained research plan in at most 600 words."
    )
    assert not is_template_copy_solution(
        "Collect baseline data, run ablations, and report uncertainty estimates."
    )


def test_find_solution_content_span_ignores_think_embedded_solution():
    text = (
        "<think>scratch <solution>fake plan</solution></think>\n"
        "<solution>  final plan text  </solution>"
    )
    span = find_solution_content_span(text)
    assert span is not None
    start, end = span
    assert text[start:end] == "final plan text"


def test_find_solution_content_span_returns_none_without_valid_solution():
    assert find_solution_content_span("No solution here") is None
    assert find_solution_content_span("<think><solution>fake</solution></think>") is None


def test_normalize_policy_output_for_grader_repairs_solution_closure():
    repaired = normalize_policy_output_for_grader("<solution>draft plan")
    assert repaired.endswith("</solution>")
    assert "<solution>" in repaired


def test_normalize_policy_output_for_grader_injects_solution_block_if_missing():
    repaired = normalize_policy_output_for_grader("plain text")
    assert "<solution>" in repaired
    assert "</solution>" in repaired


def test_check_format_compliance_missing_or_too_long():
    assert not check_format_compliance("No solution tags", max_words=750)
    long_solution = "<solution>" + ("word " * 751) + "</solution>"
    assert not check_format_compliance(long_solution, max_words=750)
    short_solution = "<solution>" + ("word " * 20) + "</solution>"
    assert check_format_compliance(short_solution, max_words=750)


def test_compute_final_reward_paper_mode_noncompliant():
    config = Config(reward_mode="paper", paper_format_penalty=1.0)
    reward, penalty = compute_final_reward(
        rubric_score=0.8,
        is_compliant=False,
        solution_word_count=200,
        config=config,
    )
    assert penalty == 1.0
    assert reward == pytest.approx(-0.2)


def test_loose_xml_fallback_scores_with_enough_levels():
    xml_text = (
        "<rubric><item num=1>"
        "<level>3</level><level>3</level><level>2</level><level>2</level>"
        "<level>1</level><level>1</level><level>0</level>"
        "</item></rubric>"
    )
    score, found_levels = compute_rubric_reward_from_loose_levels(
        xml_text,
        min_total_levels=7,
    )
    assert found_levels == 7
    assert score == pytest.approx((1.0 + 1.0 + 0.6 + 0.6 + 0.2 + 0.2 + 0.0) / 7.0)


def test_loose_xml_fallback_requires_min_levels():
    xml_text = "<level>3</level><level>2</level><level>1</level>"
    score, found_levels = compute_rubric_reward_from_loose_levels(
        xml_text,
        min_total_levels=7,
    )
    assert found_levels == 3
    assert score is None


def test_compute_think_penalty_only_in_think_solution_mode():
    think_cfg = Config(policy_output_mode="think_solution")
    sol_cfg = Config(policy_output_mode="solution_only")
    assert compute_think_penalty(think_cfg.max_think_words_soft + 40, think_cfg) > 0.0
    assert compute_think_penalty(think_cfg.max_think_words_soft - 1, think_cfg) == 0.0
    assert compute_think_penalty(999, sol_cfg) == 0.0


def test_compute_non_solution_penalty_applies_after_soft_limit():
    cfg = Config()
    assert compute_non_solution_penalty(cfg.max_non_solution_words_soft - 1, cfg) == 0.0
    assert compute_non_solution_penalty(cfg.max_non_solution_words_soft + 60, cfg) > 0.0


def test_should_drop_think_too_long_modes():
    disabled_cfg = Config(
        policy_output_mode="think_solution",
        think_too_long_drop_mode="disabled",
    )
    hard_cfg = Config(
        policy_output_mode="think_solution",
        think_too_long_drop_mode="hard",
    )
    assert not should_drop_think_too_long(
        config=disabled_cfg,
        current_think_too_long_drops=999,
        num_trainability_checks=999,
    )
    assert should_drop_think_too_long(
        config=hard_cfg,
        current_think_too_long_drops=0,
        num_trainability_checks=0,
    )


def test_should_drop_think_too_long_adaptive_cap():
    adaptive_cfg = Config(
        policy_output_mode="think_solution",
        think_too_long_drop_mode="adaptive",
        think_too_long_max_drop_rate=0.2,
    )
    assert should_drop_think_too_long(
        config=adaptive_cfg,
        current_think_too_long_drops=0,
        num_trainability_checks=4,
    )
    assert not should_drop_think_too_long(
        config=adaptive_cfg,
        current_think_too_long_drops=2,
        num_trainability_checks=8,
    )


def test_should_drop_non_solution_too_long_adaptive_cap():
    adaptive_cfg = Config(
        non_solution_too_long_drop_mode="adaptive",
        non_solution_too_long_max_drop_rate=0.1,
    )
    assert should_drop_non_solution_too_long(
        config=adaptive_cfg,
        current_non_solution_too_long_drops=0,
        num_trainability_checks=9,
    )
    assert not should_drop_non_solution_too_long(
        config=adaptive_cfg,
        current_non_solution_too_long_drops=2,
        num_trainability_checks=8,
    )


def test_group_advantages_without_normalization():
    rewards = [0.8, 0.5, 0.2]
    advantages = compute_group_advantages(rewards, normalize=False)
    mean_r = sum(rewards) / len(rewards)
    expected = [r - mean_r for r in rewards]
    assert advantages == pytest.approx(expected)


def test_noncompliant_samples_dropped_when_enabled():
    trainable = is_sample_trainable(
        plan_text="<solution>short plan</solution>",
        solution_word_count=2,
        is_compliant=False,
        sample_logprobs=[-1.0, -0.9],
        drop_noncompliant_samples=True,
        min_words=1,
    )
    assert trainable is False


def test_noncompliant_samples_allowed_when_disabled():
    trainable = is_sample_trainable(
        plan_text="<solution>short plan</solution>",
        solution_word_count=2,
        is_compliant=False,
        sample_logprobs=[-1.0, -0.9],
        drop_noncompliant_samples=False,
        min_words=1,
    )
    assert trainable is True


def test_trainability_drop_reason_priority_and_categories():
    assert (
        get_trainability_drop_reason(
            plan_text="<solution>ok</solution>",
            solution_word_count=2,
            is_compliant=False,
            sample_logprobs=[-1.0],
            drop_noncompliant_samples=True,
            min_words=1,
        )
        == "format_noncompliant"
    )
    assert (
        get_trainability_drop_reason(
            plan_text="<solution>ok</solution>",
            solution_word_count=2,
            is_compliant=True,
            sample_logprobs=[-1.0],
            drop_noncompliant_samples=True,
            min_words=10,
        )
        == "too_short_solution"
    )
    assert (
        get_trainability_drop_reason(
            plan_text="<solution>ok</solution>",
            solution_word_count=20,
            is_compliant=True,
            sample_logprobs=None,
            drop_noncompliant_samples=True,
            min_words=10,
        )
        == "missing_logprobs"
    )


def test_trainability_drop_reason_think_too_long():
    assert (
        get_trainability_drop_reason(
            plan_text="<think>...</think><solution>ok</solution>",
            solution_word_count=100,
            think_word_count=220,
            max_think_words_hard=180,
            is_compliant=True,
            sample_logprobs=[-1.0],
            drop_noncompliant_samples=True,
            min_words=10,
        )
        == "think_too_long"
    )


def test_trainability_drop_reason_non_solution_too_long():
    assert (
        get_trainability_drop_reason(
            plan_text="<think>...</think><solution>ok</solution>outside words",
            solution_word_count=100,
            non_solution_word_count=320,
            max_non_solution_words_hard=260,
            is_compliant=True,
            sample_logprobs=[-1.0],
            drop_noncompliant_samples=True,
            min_words=10,
        )
        == "non_solution_too_long"
    )


def test_trainability_drop_reason_template_copy_solution():
    placeholder = "A complete and self-contained research plan in at most 750 words."
    assert (
        get_trainability_drop_reason(
            plan_text=f"<solution>{placeholder}</solution>",
            solution_word_count=11,
            solution_text=placeholder,
            is_compliant=True,
            sample_logprobs=[-1.0],
            drop_noncompliant_samples=True,
            drop_template_copy_solutions=True,
            min_words=30,
        )
        == "template_copy_solution"
    )


def test_stabilization_defaults_solution_only_and_target_length():
    config = Config()
    assert config.policy_output_mode == "solution_only"
    assert config.target_word_count == 600
    assert config.min_words == 180
    assert config.metrics_mode == "compact"


def test_sdpo_success_cutoff_fixed_mode():
    cutoff, source = compute_sdpo_success_cutoff(
        history_batches_scores=[[0.2, 0.5]],
        current_batch_scores=[0.7, 0.8],
        mode="fixed",
        adaptive_quantile=0.9,
        min_samples=10,
        use_current_batch_bootstrap=True,
        floor=0.0,
        ceiling=1.0,
        fixed_threshold=0.77,
    )
    assert cutoff == pytest.approx(0.77)
    assert source == "fixed"


def test_sdpo_success_cutoff_adaptive_history_quantile():
    cutoff, source = compute_sdpo_success_cutoff(
        history_batches_scores=[[0.1, 0.2, 0.3, 0.9]],
        current_batch_scores=[],
        mode="adaptive_quantile",
        adaptive_quantile=0.75,
        min_samples=1,
        use_current_batch_bootstrap=True,
        floor=0.0,
        ceiling=1.0,
        fixed_threshold=0.85,
    )
    assert cutoff == pytest.approx(0.45)
    assert source == "history"


def test_sdpo_success_cutoff_bootstraps_with_current_batch():
    cutoff, source = compute_sdpo_success_cutoff(
        history_batches_scores=[[0.2]],
        current_batch_scores=[0.4, 0.8, 0.9],
        mode="adaptive_quantile",
        adaptive_quantile=0.9,
        min_samples=16,
        use_current_batch_bootstrap=True,
        floor=0.0,
        ceiling=1.0,
        fixed_threshold=0.85,
    )
    assert cutoff == pytest.approx(0.87)
    assert source == "history+current_bootstrap"


def test_sdpo_success_cutoff_no_data_respects_bounds():
    cutoff, source = compute_sdpo_success_cutoff(
        history_batches_scores=[],
        current_batch_scores=[],
        mode="adaptive_quantile",
        adaptive_quantile=0.9,
        min_samples=16,
        use_current_batch_bootstrap=True,
        floor=0.0,
        ceiling=0.85,
        fixed_threshold=0.85,
    )
    assert cutoff == pytest.approx(0.85)
    assert source == "no_data"
