"""Text-to-Speech Test Script for Jarvis (Phase 2).

This script tests the TTS system:
1. Speaks a standard sentence using the primary edge-tts engine.
2. Speaks a sentence containing markdown to verify markdown stripping.
3. Speaks a conversational sentence.
4. Directly tests the offline pyttsx3 fallback engine.
"""

from tts import speak, speak_fallback, strip_markdown


def main() -> None:
    """Run all Text-to-Speech verification tests."""
    print("=" * 60)
    print("  Jarvis Phase 2: Text-to-Speech Test (edge-tts + pyttsx3)")
    print("=" * 60)
    print("Listen to your speakers or headphones for Jarvis speaking.\n")

    # Test 1: Standard sentence with primary engine
    sentence_1 = "Hello! I am Jarvis, your personal AI assistant. All audio systems are functioning normally."
    print("--- Test 1: Primary Engine (edge-tts) ---")
    print(f"Speaking: \"{sentence_1}\"")
    speak(sentence_1)
    print("[OK] Test 1 finished.\n")

    # Test 2: Sentence with markdown formatting
    sentence_2 = (
        "Here is a **critical** reminder: `#1 priority` is *ready* for "
        "[deployment](https://example.com) on your system!"
    )
    print("--- Test 2: Markdown Stripping ---")
    print(f"Raw Input: \"{sentence_2}\"")
    print(f"Stripped:  \"{strip_markdown(sentence_2)}\"")
    print("Speaking stripped version...")
    speak(sentence_2)
    print("[OK] Test 2 finished.\n")

    # Test 3: Conversational sentence
    sentence_3 = "How may I be of service to you today, sir?"
    print("--- Test 3: Conversational Sentence ---")
    print(f"Speaking: \"{sentence_3}\"")
    speak(sentence_3)
    print("[OK] Test 3 finished.\n")

    # Test 4: Direct test of offline fallback engine
    fallback_sentence = "This is a direct test of the offline fallback speech engine."
    print("--- Test 4: Offline Fallback Engine (pyttsx3) ---")
    print(f"Speaking directly via fallback: \"{fallback_sentence}\"")
    speak_fallback(fallback_sentence)
    print("[OK] Test 4 finished.\n")

    print("=" * 60)
    print("All Text-to-Speech tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
