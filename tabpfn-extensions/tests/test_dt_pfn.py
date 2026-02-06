"""Tests for the TabPFN-based Decision Tree implementation.

This file tests the Decision Tree TabPFN implementations in tabpfn_extensions.rf_pfn.
"""

from __future__ import annotations

import pytest

from tabpfn_extensions.rf_pfn.sklearn_based_decision_tree_tabpfn import (
    DecisionTreeTabPFNClassifier,
    DecisionTreeTabPFNRegressor,
)
from test_base_tabpfn import BaseClassifierTests, BaseRegressorTests


class TestDecisionTreeClassifier(BaseClassifierTests):
    """Test DecisionTreeTabPFNClassifier using the BaseClassifierTests framework."""

    @pytest.fixture
    def estimator(self, tabpfn_classifier):
        """Provide a TabPFN-based DecisionTreeClassifier as the estimator."""
        return DecisionTreeTabPFNClassifier(
            tabpfn=tabpfn_classifier,
            max_depth=3,  # Shallow tree for speed
            random_state=42,
        )

    @pytest.mark.skip(reason="DecisionTreeTabPFN doesn't fully support text features")
    def test_with_text_features(self, estimator, dataset_generator):
        pass

    @pytest.mark.skip(
        reason="DecisionTreeTabPFN needs additional work to pass all sklearn estimator checks",
    )
    def test_passes_estimator_checks(self, estimator):
        pass


class TestAdaptiveDecisionTreeClassifier(TestDecisionTreeClassifier):
    @pytest.fixture
    def estimator(self, tabpfn_classifier):
        """Provide a TabPFN-based DecisionTreeClassifier as the estimator."""
        return DecisionTreeTabPFNClassifier(
            tabpfn=tabpfn_classifier,
            max_depth=3,  # Shallow tree for speed
            random_state=42,
            adaptive_tree=True,
        )


class TestDecisionTreeRegressor(BaseRegressorTests):
    """Test DecisionTreeTabPFNRegressor using the BaseRegressorTests framework."""

    @pytest.fixture
    def estimator(self, tabpfn_regressor):
        """Provide a TabPFN-based DecisionTreeRegressor as the estimator."""
        return DecisionTreeTabPFNRegressor(
            tabpfn=tabpfn_regressor,
            max_depth=2,  # Shallow tree for speed
            random_state=42,
        )

    @pytest.mark.skip(reason="DecisionTreeTabPFN doesn't fully support text features")
    def test_with_text_features(self, estimator, dataset_generator):
        pass

    @pytest.mark.skip(
        reason="DecisionTreeTabPFN needs additional work to pass all sklearn estimator checks",
    )
    def test_passes_estimator_checks(self, estimator):
        pass


class TestAdaptiveDecisionTreeRegressor(TestDecisionTreeRegressor):
    @pytest.fixture
    def estimator(self, tabpfn_regressor):
        """Provide a TabPFN-based DecisionTreeRegressor as the estimator."""
        return DecisionTreeTabPFNRegressor(
            tabpfn=tabpfn_regressor,
            max_depth=2,  # Shallow tree for speed
            random_state=42,
            adaptive_tree=True,
        )
