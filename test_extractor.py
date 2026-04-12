#!/usr/bin/env python3
"""
Tests for ArchitecturalExtractor — dimension filtering, metadata extraction,
confidence scoring, and end-to-end PDF extraction.

Run: pytest test_extractor.py -v
"""

import pytest
from architectural_extractor import ArchitecturalExtractor


# ── Unit tests: filter_candidate_dimensions ──

class TestFilterCandidates:
    """Test that noise is removed and real dimensions survive."""

    def test_phone_numbers_removed(self):
        text = "Width 840 Tel: +250788815711 Length 670"
        candidates = ArchitecturalExtractor.filter_candidate_dimensions(text)
        assert 250788815711 not in candidates
        assert 788815711 not in candidates
        assert 840 in candidates
        assert 670 in candidates

    def test_short_phone_removed(self):
        text = "0788244592 width 840"
        candidates = ArchitecturalExtractor.filter_candidate_dimensions(text)
        assert 788244592 not in candidates
        assert 840 in candidates

    def test_dates_removed(self):
        text = "FEBRUARY 2025 width 840 length 670"
        candidates = ArchitecturalExtractor.filter_candidate_dimensions(text)
        assert 2025 not in candidates
        assert 840 in candidates

    def test_construction_specs_removed(self):
        text = "150mm thick concrete surround. Building 840 x 670"
        candidates = ArchitecturalExtractor.filter_candidate_dimensions(text)
        # 150 with "mm thick" should be removed
        assert 840 in candidates
        assert 670 in candidates

    def test_upi_reference_removed(self):
        text = "UPI:1/03/05/03/10947 width 840"
        candidates = ArchitecturalExtractor.filter_candidate_dimensions(text)
        assert 10947 not in candidates
        assert 840 in candidates

    def test_version_numbers_removed(self):
        text = "GSPublisherVersion 0.27.100.100 width 840"
        candidates = ArchitecturalExtractor.filter_candidate_dimensions(text)
        assert 840 in candidates

    def test_small_numbers_excluded(self):
        text = "4 rooms, 3 bathrooms, width 840"
        candidates = ArchitecturalExtractor.filter_candidate_dimensions(text)
        assert 4 not in candidates
        assert 3 not in candidates
        assert 840 in candidates

    def test_typical_floor_plan_dimensions(self):
        """Test with real dimension annotations from document.pdf"""
        text = "840 670 510 370 270 240 210 160 150 130 120 100 90 80 78 63 46 20"
        candidates = ArchitecturalExtractor.filter_candidate_dimensions(text)
        assert 840 in candidates
        assert 670 in candidates
        # Small numbers (< 10) excluded
        assert all(c >= 10 for c in candidates)

    def test_deduplication(self):
        text = "840 670 840 670"
        candidates = ArchitecturalExtractor.filter_candidate_dimensions(text)
        assert candidates.count(840) == 1
        assert candidates.count(670) == 1

    def test_decimal_dimensions(self):
        """document1.pdf uses meters with decimals"""
        text = "16.80 6.40 3.20 2.80 1.60 1.20 0.40"
        candidates = ArchitecturalExtractor.filter_candidate_dimensions(text)
        assert 16.80 in candidates


# ── Unit tests: extract_metadata ──

class TestExtractMetadata:

    def test_scale_colon(self):
        scale, _, _ = ArchitecturalExtractor.extract_metadata("Scale 1:65")
        assert scale == "1:65"

    def test_scale_slash(self):
        scale, _, _ = ArchitecturalExtractor.extract_metadata("SCALE: 1/80")
        assert scale == "1:80"

    def test_unit_cm_explicit(self):
        _, unit, conf = ArchitecturalExtractor.extract_metadata(
            "All dimensions (in cm) to be checked on site"
        )
        assert unit == "cm"
        assert conf == "high"

    def test_unit_meters_explicit(self):
        _, unit, conf = ArchitecturalExtractor.extract_metadata(
            "All dimensions are in meters and to be checked"
        )
        assert unit == "m"
        assert conf == "high"

    def test_no_unit(self):
        _, unit, conf = ArchitecturalExtractor.extract_metadata("Width 840 Length 670")
        assert unit is None
        assert conf == "low"


# ── Unit tests: assess_confidence ──

class TestConfidence:

    def setup_method(self):
        self.ext = ArchitecturalExtractor(verbose=False)

    def test_high_confidence_with_scale_and_unit(self):
        candidates = [840, 670, 510, 370, 270, 240, 160, 150]
        score, w, l = self.ext.assess_confidence(candidates, "1:65", "cm")
        assert score >= 6
        assert w == 840
        assert l == 670

    def test_low_confidence_few_candidates(self):
        candidates = [840]
        score, _, _ = self.ext.assess_confidence(candidates, None, None)
        assert score == 0

    def test_moderate_confidence_no_unit(self):
        candidates = [840, 670, 510]
        score, w, l = self.ext.assess_confidence(candidates, "1:65", None)
        # Scale adds 2, but no unit means lower score
        assert w == 840
        assert l == 670

    def test_area_sanity_check(self):
        """840cm x 670cm = 56.28 m2 — typical residential, should pass."""
        candidates = [840, 670]
        score, _, _ = self.ext.assess_confidence(candidates, "1:65", "cm")
        assert score >= 6


# ── Unit tests: calculate_area_m2 ──

class TestAreaCalc:

    def test_cm_to_m2(self):
        area = ArchitecturalExtractor.calculate_area_m2(840, 670, "cm")
        assert abs(area - 56.28) < 0.01

    def test_m_to_m2(self):
        area = ArchitecturalExtractor.calculate_area_m2(16.80, 6.40, "m")
        assert abs(area - 107.52) < 0.01

    def test_mm_to_m2(self):
        area = ArchitecturalExtractor.calculate_area_m2(8400, 6700, "mm")
        assert abs(area - 56.28) < 0.01

    def test_none_inputs(self):
        assert ArchitecturalExtractor.calculate_area_m2(None, 670, "cm") is None


# ── Integration tests: full pipeline on real PDFs ──

class TestFullPipeline:
    """End-to-end tests using the sample floor plan PDFs.
    These tests require LiteParse to be installed.
    """

    @pytest.fixture
    def extractor(self):
        return ArchitecturalExtractor(verbose=False)

    @pytest.mark.skipif(
        not ArchitecturalExtractor(verbose=False)._liteparse,
        reason="LiteParse not installed"
    )
    def test_document_pdf_liteparse_extraction(self, extractor):
        """document.pdf: Rwandan residential floor plan, dimensions in cm."""
        text = extractor.extract_with_liteparse(
            "samples/document.pdf"
        )
        assert text is not None
        assert len(text) > 100

        # Should contain key dimension numbers
        candidates = extractor.filter_candidate_dimensions(text)
        assert 840 in candidates
        assert 670 in candidates

        # Should find scale and unit
        scale, unit, conf = extractor.extract_metadata(text)
        assert scale == "1:65"
        assert unit == "cm"
        assert conf == "high"

    @pytest.mark.skipif(
        not ArchitecturalExtractor(verbose=False)._liteparse,
        reason="LiteParse not installed"
    )
    def test_document1_pdf_liteparse_extraction(self, extractor):
        """document1.pdf: Ground floor plan, dimensions in meters."""
        text = extractor.extract_with_liteparse(
            "samples/document1.pdf"
        )
        assert text is not None
        assert len(text) > 100

        # Should find scale
        scale, unit, conf = extractor.extract_metadata(text)
        assert scale is not None  # 1:80

    def test_parse_dimensions_document_text(self, extractor):
        """Simulate document.pdf text through parse_dimensions_from_text."""
        # Representative text from LiteParse output
        text = """
        REPUBLIC OF RWANDA
        PROPOSED RESIDENTIAL HOUSE
        Tel: +250788815711
        Tel:+0788244592
        UPI:1/03/05/03/10947
        All dimensions (in cm) to be checked on site.
        150mm thick concrete surround
        FEBRUARY 2025
        840 670 510 370 270 240 210 160 150 130 120 100 90 80
        Bed room 1 Bed room 2 Kitchen Corridor Bathroom
        Living room Master bed room
        1:65
        GSPublisherVersion 0.27.100.100
        """
        result = extractor.parse_dimensions_from_text(text)
        assert result is not None
        assert result['width'] == 840
        assert result['length'] == 670
        assert result['unit'] == 'cm'
        assert result['scale'] == '1:65'

        area = extractor.calculate_area_m2(result['width'], result['length'], result['unit'])
        assert 50 < area < 60  # 840 * 670 / 10000 = 56.28 m2
