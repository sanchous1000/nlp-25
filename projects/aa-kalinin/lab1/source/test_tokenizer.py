from nlp import Tokenizer, SentenceSegmenter, TextProcessor


def test_email_tokenization():
    """Test that email addresses are kept as single tokens."""
    print("=" * 60)
    print("TEST: Email Address Tokenization")
    print("=" * 60)

    tokenizer = Tokenizer()
    test_cases = [
        "Contact us at support@example.com for help.",
        "Send your resume to john.doe@company.org",
        "Multiple emails: alice@test.com and bob@test.net",
    ]

    for text in test_cases:
        tokens = tokenizer.tokenize(text)
        print(f"\nInput: {text}")
        print(f"Tokens: {tokens}")

        email_preserved = any('@' in t and '.' in t.split('@')[-1] for t in tokens)
        print(f"✓ Email preserved as single token: {email_preserved}")


def test_url_tokenization():
    """Test that URLs are handled appropriately."""
    print("\n" + "=" * 60)
    print("TEST: URL Tokenization")
    print("=" * 60)

    tokenizer = Tokenizer()
    test_cases = [
        "Visit SPACE.com for news.",
        "Check out Forbes.com - it's great!",
        "News from Reuters.com today.",
    ]

    for text in test_cases:
        tokens = tokenizer.tokenize(text)
        print(f"\nInput: {text}")
        print(f"Tokens: {tokens}")


def test_phone_tokenization():
    """Test that phone numbers are kept as single tokens."""
    print("\n" + "=" * 60)
    print("TEST: Phone Number Tokenization (US Format)")
    print("=" * 60)

    tokenizer = Tokenizer()
    test_cases = [
        "Call us at 1-800-555-1234 for support.",
        "Phone: (555) 123-4567 or 555-789-0123",
        "Reach me at +1 202-555-0147 anytime.",
        "Customer service: 800-CALL-NOW",
    ]

    for text in test_cases:
        tokens = tokenizer.tokenize(text)
        print(f"\nInput: {text}")
        print(f"Tokens: {tokens}")


def test_abbreviation_tokenization():
    """Test that abbreviations are handled correctly."""
    print("\n" + "=" * 60)
    print("TEST: Abbreviation Tokenization")
    print("=" * 60)

    tokenizer = Tokenizer()
    segmenter = SentenceSegmenter()
    test_cases = [
        "Dr. Smith visited St. Louis.",
        "Mr. and Mrs. Johnson live on Oak Ave.",
        "The company (Apple Inc.) is located in the U.S.",
        "Prof. Williams teaches at M.I.T.",
    ]

    for text in test_cases:
        tokens = tokenizer.tokenize(text)
        sentences = segmenter.segment(text)
        print(f"\nInput: {text}")
        print(f"Tokens: {tokens}")
        print(f"Sentences: {sentences}")


def test_emoticon_tokenization():
    """Test that emoticons are kept as single tokens."""
    print("\n" + "=" * 60)
    print("TEST: Emoticon Tokenization")
    print("=" * 60)

    tokenizer = Tokenizer()
    test_cases = [
        "That's great :) I love it!",
        "So sad :( but we'll manage.",
        "Check this out ;-) It's cool!",
        "I'm happy ^_^ about this.",
        "Love you <3",
    ]

    for text in test_cases:
        tokens = tokenizer.tokenize(text)
        print(f"\nInput: {text}")
        print(f"Tokens: {tokens}")


def test_currency_tokenization():
    """Test that currency amounts are handled correctly."""
    print("\n" + "=" * 60)
    print("TEST: Currency Tokenization")
    print("=" * 60)

    tokenizer = Tokenizer()
    test_cases = [
        "The price is $50.99 per unit.",
        "Revenue reached $1.5 billion this quarter.",
        "Costs were €25,000 for the project.",
        "Stock fell $36.50 in trading.",
    ]

    for text in test_cases:
        tokens = tokenizer.tokenize(text)
        print(f"\nInput: {text}")
        print(f"Tokens: {tokens}")


def test_contraction_tokenization():
    """Test that contractions are handled correctly."""
    print("\n" + "=" * 60)
    print("TEST: Contraction Tokenization")
    print("=" * 60)

    tokenizer = Tokenizer()
    test_cases = [
        "I can't believe it's already Friday.",
        "They're going to the store.",
        "We've been waiting for hours.",
        "She wouldn't let him go.",
        "It's John's birthday today.",
    ]

    for text in test_cases:
        tokens = tokenizer.tokenize(text)
        print(f"\nInput: {text}")
        print(f"Tokens: {tokens}")


def test_hashtag_mention_tokenization():
    """Test that hashtags and mentions are kept as single tokens."""
    print("\n" + "=" * 60)
    print("TEST: Hashtag and Mention Tokenization")
    print("=" * 60)

    tokenizer = Tokenizer()
    test_cases = [
        "Check out #MachineLearning for more info.",
        "Thanks @johndoe for the help!",
        "Trending: #AI #DeepLearning #NLP",
    ]

    for text in test_cases:
        tokens = tokenizer.tokenize(text)
        print(f"\nInput: {text}")
        print(f"Tokens: {tokens}")


def test_full_pipeline():
    """Test the complete text processing pipeline."""
    print("\n" + "=" * 60)
    print("TEST: Full Pipeline (Token, Stem, Lemma)")
    print("=" * 60)

    processor = TextProcessor()
    test_cases = [
        "The companies are running successfully.",
        "Dr. Smith's patients were being treated carefully.",
        "Stock prices fell sharply after the announcement.",
    ]

    for text in test_cases:
        results = processor.process_text(text)
        print(f"\nInput: {text}")
        print(f"{'Token':<20} {'Stem':<15} {'Lemma':<15}")
        print("-" * 50)
        for sentence in results:
            for token, stem, lemma in sentence:
                print(f"{token:<20} {stem:<15} {lemma:<15}")
            print()


def test_sentence_segmentation():
    """Test sentence segmentation with edge cases."""
    print("\n" + "=" * 60)
    print("TEST: Sentence Segmentation")
    print("=" * 60)

    segmenter = SentenceSegmenter()
    test_cases = [
        "Dr. Smith visited the U.S. last week. He enjoyed it.",
        "The price was $50.00. That's expensive!",
        "Are you ready? I hope so. Let's go!",
        "Mr. and Mrs. Johnson arrived at 5 p.m. They were late.",
    ]

    for text in test_cases:
        sentences = segmenter.segment(text)
        print(f"\nInput: {text}")
        print(f"Sentences ({len(sentences)}):")
        for i, sent in enumerate(sentences, 1):
            print(f"  {i}. {sent}")


def main():
    """Run all tokenization tests."""
    print("\n" + "=" * 60)
    print("       TOKENIZER TEST SUITE")
    print("=" * 60)

    test_email_tokenization()
    test_url_tokenization()
    test_phone_tokenization()
    test_abbreviation_tokenization()
    test_emoticon_tokenization()
    test_currency_tokenization()
    test_contraction_tokenization()
    test_hashtag_mention_tokenization()
    test_sentence_segmentation()
    test_full_pipeline()

    print("\n" + "=" * 60)
    print("       ALL TESTS COMPLETED")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
