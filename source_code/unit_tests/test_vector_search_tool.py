import logging
from pathlib import Path
from tool_management.tools.vector_search_tool import VectorSearchTool


def test_vector_search():
    desktop = Path.home() / "Desktop"

    input_files = [
        desktop / "2025 SS Topics Bachelor Thesis LUX Marian_2.pdf",
        desktop / "firmenbuchpublizitaet_beispiele.docx"
    ]

    for path in input_files:
        if not path.exists():
            print(f"Missing file: {path}")
            return

    print("Initialising VectorSearchTool with multiple documents")
    tool = VectorSearchTool(input_files)

    queries = [
        "Develop a ReAct (Reason + Act) inspired Agent",
        "Was bedeutet negative Publizität im Firmenbuch?"
    ]

    for query in queries:
        print(f"\n Query: {query}\n")
        response = tool.execute(query)

        print("✅ ToolResponse:")
        print("Success:", response.success)
        print("Content:\n", response.content)
        print("Metadata:", response.metadata)
        print("-" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_vector_search()
