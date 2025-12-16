from .infer import load_best_model, predict_one


def main():
    print("Movie Genre Classifier Demo")
    print("Type a movie description (or type 'q' to quit).")

    model = load_best_model()

    while True:
        text = input("\nEnter description: ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            break
        if not text:
            print("Please type a non-empty description.")
            continue

        label, conf = predict_one(model, text)

        if conf is None:
            print(f"Predicted genre: {label}")
        else:
            print(f"Predicted genre: {label} (confidence: {conf:.3f})")


if __name__ == "__main__":
    main()
