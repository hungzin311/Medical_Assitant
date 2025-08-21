from agents.agent_decision import create_agent_graph

def main(): 
    graph = create_agent_graph()
    try:
        image_bytes = graph.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(image_bytes)
    except Exception:
        # This requires some extra dependencies and is optional
        pass

if __name__ == "__main__":
    main()
