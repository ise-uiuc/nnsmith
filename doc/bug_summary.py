# pip install marko
import marko
import marko.inline
from rich.console import Console
from rich.table import Table
from tomark import Tomark

if __name__ == "__main__":
    text = open("bugs.md").read()
    doc = marko.parse(text)

    systems = [
        "PyTorch",
        "PyTorch-ONNX Converter",
        "ONNX",
        "ONNXRuntime",
        "TVM",
        "TensorRT",
        "TensorFlow",
        "Hidet",
    ]

    table = Table(title="Bug Summary")
    table.add_column("System", justify="right", style="cyan", no_wrap=True)
    table.add_column("#Fixed", style="magenta")
    table.add_column("#Confirmed", style="magenta")
    table.add_column("#Pending", style="magenta")
    table.add_column("#Total", justify="right", style="green")

    # Level-2 headings must be:
    #    - One of the system names
    #    - Url'ed
    bugs = {}
    system = None
    for child in doc.children:
        if isinstance(child, marko.block.Heading):
            if child.level == 2:
                assert len(child.children) == 1
                url = child.children[0]
                assert isinstance(
                    url, marko.inline.Link
                ), "Level-2 headings must be url'ed to the system homepage"
                text: marko.inline.RawText = url.children[0]
                system = text.children
                assert system in systems, f"Parsed name: {system} not in {systems}"
        elif isinstance(child, marko.block.List):
            if system is None:
                continue

            bugs[system] = {"#Fixed": 0, "#Confirmed": 0, "#Pending": 0, "#Total": 0}
            for item in child.children:
                url = item.children[0].children[-1]
                assert isinstance(
                    url, marko.inline.Link
                ), f"No link found in the list {item.children[0].children}"

                text = item.children[0].children[0]
                if isinstance(text, marko.inline.RawText):
                    text = text.children
                    if "✅" in text:
                        bugs[system]["#Fixed"] += 1
                    elif "🔵" in text:
                        bugs[system]["#Confirmed"] += 1
                    else:
                        bugs[system]["#Pending"] += 1
                else:
                    bugs[system]["#Pending"] += 1

                bugs[system]["#Total"] += 1
            system = None

    for system, item in bugs.items():
        table.add_row(
            system,
            str(item["#Fixed"]),
            str(item["#Confirmed"]),
            str(item["#Pending"]),
            str(item["#Total"]),
        )

    # Add a total row
    table.add_row(
        "Sum",
        str(sum(item["#Fixed"] for item in bugs.values())),
        str(sum(item["#Confirmed"] for item in bugs.values())),
        str(sum(item["#Pending"] for item in bugs.values())),
        str(sum(item["#Total"] for item in bugs.values())),
        style="on green",
    )

    # Create a markdown table
    md_table = [{"System": k, **v} for k, v in bugs.items()]
    md_table.append(
        {
            "System": "Sum",
            "#Fixed": sum(item["#Fixed"] for item in bugs.values()),
            "#Confirmed": sum(item["#Confirmed"] for item in bugs.values()),
            "#Pending": sum(item["#Pending"] for item in bugs.values()),
            "#Total": sum(item["#Total"] for item in bugs.values()),
        }
    )

    # TODO(@ganler): consistency check
    print(Tomark.table(md_table))

    console = Console()
    console.print(table)
