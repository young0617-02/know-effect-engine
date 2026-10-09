import pytest

from kee.retrieval.rbac import AccessContext, UnsafeIdentifier, build_filter_expr


def test_role_default_level():
    expr = build_filter_expr(AccessContext(role="newcomer"))
    assert "security_level <= 1" in expr


def test_industry_and_project_filters():
    ctx = AccessContext(role="delivery_engineer", industries=["金融"], projects=["PRJ-FIN-2024-017"])
    expr = build_filter_expr(ctx)
    assert 'industry in ["金融"]' in expr
    assert 'project in ["PRJ-FIN-2024-017"]' in expr
    assert "security_level <= 2" in expr


def test_admin_scope_skips_industry_project():
    expr = build_filter_expr(AccessContext(role="knowledge_owner", industries=["金融"], projects=["P1"]))
    assert "industry" not in expr and "project" not in expr
    assert "security_level <= 3" in expr


def test_override_level_wins_over_role():
    ctx = AccessContext(role="newcomer", max_security_level=3)
    assert "security_level <= 3" in build_filter_expr(ctx)


@pytest.mark.parametrize(
    "bad",
    [
        '金融"] or security_level >= 3',
        'a" == "b',
        'a" or "1"=="1',
        "",
        "x;y",
        "金融' OR '1'='1",
    ],
)
def test_expression_injection_is_rejected(bad):
    ctx = AccessContext(role="delivery_engineer", industries=[bad])
    with pytest.raises(UnsafeIdentifier):
        build_filter_expr(ctx)
