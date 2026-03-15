from pipelines import handle_query

def main():
    print("Mini Transformer NLP Lab")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Query: ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        result = handle_query(query)
        print("\nResult:")
        print(result)
        print("-" * 60)

if __name__ == "__main__":
    main()