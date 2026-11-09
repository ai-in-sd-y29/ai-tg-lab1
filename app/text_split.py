"""Разбиение длинного текста на сообщения Telegram.

Telegram считает длину сообщения в кодовых единицах UTF-16, поэтому эмодзи
и другие символы вне BMP занимают две позиции из 4096.
"""

TELEGRAM_LIMIT = 4096


def utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Делит текст на части не длиннее limit без потери символов.

    Разрез ищется по абзацу, затем по строке, затем по пробелу; если подходящей
    границы нет, слово режется по символам. Склейка частей даёт исходный текст,
    кроме частей из одних пробелов: их отправить нельзя, и они пропускаются.
    """
    if limit < 2:
        raise ValueError("limit должен быть не меньше 2")
    parts: list[str] = []
    rest = text
    while rest:
        if utf16_len(rest) <= limit:
            parts.append(rest)
            break
        # Самый длинный префикс, укладывающийся в лимит.
        end, size = 0, 0
        for char in rest:
            char_size = 2 if ord(char) > 0xFFFF else 1
            if size + char_size > limit:
                break
            size += char_size
            end += 1
        window = rest[:end]
        cut = end
        for separator in ("\n\n", "\n", " "):
            position = window.rfind(separator)
            # Не режем слишком близко к началу, чтобы не плодить крошечные части.
            if position > end // 2:
                cut = position + len(separator)
                break
        parts.append(rest[:cut])
        rest = rest[cut:]
    return [part for part in parts if part.strip()]
