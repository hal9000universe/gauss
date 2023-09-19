from src.data_engine.data_pipe import get_tokenizer, format_decoding


def test_tokenizer():
    # get tokenizer
    tokenizer = get_tokenizer()
    original = ("-312.33953857421875*x_(0) + 856.2227783203125*x_(1) + 352.9467468261719*x_(2) + 529.22119140625*x_(3) "
                "= -435.47772216796875 /n789.8570556640625*x_(0) + 19.7451114654541*x_(1) + 642.3790283203125*x_(2) + "
                "339.9715576171875*x_(3) = -719.56396484375 /n309.1565246582031*x_(0) + 782.404296875*x_(1) + "
                "-584.4415283203125*x_(2) + 984.8065185546875*x_(3) = -891.5484008789062 /n816.6369018554688*x_(0) + "
                "-201.58993530273438*x_(1) + 488.4422912597656*x_(2) + -371.6492614746094*x_(3) = "
                "-934.2286376953125?/swap_rows(0,3)")
    encoded = tokenizer.encode(original)
    decoded = format_decoding(tokenizer.decode(encoded.ids))
    assert original == decoded
