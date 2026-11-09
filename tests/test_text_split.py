import pytest

from app.text_split import TELEGRAM_LIMIT, split_message, utf16_len


def test_short_text_is_single_part():
    # Act / Assert
    assert split_message("Привет") == ["Привет"]


def test_empty_or_blank_text_gives_no_parts():
    # Act / Assert
    assert split_message("") == []
    assert split_message("   \n ") == []


def test_exact_limit_is_not_split():
    # Arrange
    text = "я" * TELEGRAM_LIMIT
    # Act / Assert
    assert split_message(text) == [text]


def test_split_prefers_paragraph_boundary():
    # Arrange
    first = "а" * 3000
    second = "б" * 3000
    # Act
    parts = split_message(first + "\n\n" + second)
    # Assert
    assert parts == [first + "\n\n", second]


def test_word_without_spaces_is_cut_by_characters():
    # Arrange
    text = "x" * 10000
    # Act
    parts = split_message(text)
    # Assert
    assert [len(p) for p in parts] == [4096, 4096, 1808]
    assert "".join(parts) == text


def test_emoji_count_as_two_utf16_units():
    # Arrange
    text = "😀" * 3000  # 6000 единиц UTF-16
    # Act
    parts = split_message(text)
    # Assert
    assert all(utf16_len(p) <= TELEGRAM_LIMIT for p in parts)
    assert "".join(parts) == text
    assert len(parts) == 2


@pytest.mark.parametrize("limit", [10, 37, 100])
def test_order_and_content_preserved_for_small_limits(limit):
    # Arrange
    text = " ".join(f"слово{i}" for i in range(200))
    # Act
    parts = split_message(text, limit=limit)
    # Assert
    assert "".join(parts) == text
    assert all(0 < utf16_len(p) <= limit for p in parts)
