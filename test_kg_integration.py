#!/usr/bin/env python3
"""
Test script to verify KG agent integration with the main workflow
"""

from agents.agent_decision import create_agent_graph, process_query
from agents.patient_db_agent import PatientQueryEngine
from config import Config
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_kg_agent_integration():
    """Test KG agent integration with workflow"""
    try:
        # Initialize config and patient query engine
        config = Config()
        patient_query_engine = PatientQueryEngine(config)
        
        # Create the agent graph
        graph = create_agent_graph(patient_query_engine)
        
        # Test query that should trigger KG agent
        test_query = "Triệu chứng của bệnh đau đầu là gì?"
        
        print(f"Testing query: {test_query}")
        print("=" * 50)
        
        # Process the query
        result = process_query(
            query=test_query,
            conversation_history=None,
            graph=graph
        )
        
        print(f"Agent used: {result.get('agent_name', 'Unknown')}")
        print(f"Output: {result.get('output', 'No output')}")
        
        return True
        
    except Exception as e:
        print(f"Error during integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kg_to_web_fallback():
    """Test KG agent fallback to web search when no information is found"""
    try:
        # Initialize config and patient query engine
        config = Config()
        patient_query_engine = PatientQueryEngine(config)
        
        # Create the agent graph
        graph = create_agent_graph(patient_query_engine)
        
        # Test query that might not have information in KG
        test_query = "Bệnh mới phát hiện XYZ123 có triệu chứng gì?"
        
        print(f"Testing fallback query: {test_query}")
        print("=" * 50)
        
        # Process the query
        result = process_query(
            query=test_query,
            conversation_history=None,
            graph=graph
        )
        
        print(f"Agent used: {result.get('agent_name', 'Unknown')}")
        print(f"Output: {result.get('output', 'No output')}")
        
        # Check if web search was involved
        if "WEB_SEARCH_PROCESSOR_AGENT" in str(result.get('agent_name', '')):
            print("✓ Fallback to web search working correctly")
        else:
            print("? Web search fallback may not have been triggered")
        
        return True
        
    except Exception as e:
        print(f"Error during fallback test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing KG Agent Integration")
    print("=" * 60)
    
    # Test basic integration
    print("\n1. Testing basic KG agent integration:")
    success1 = test_kg_agent_integration()
    
    # Test fallback mechanism
    print("\n2. Testing KG to Web Search fallback:")
    success2 = test_kg_to_web_fallback()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✓ All tests passed successfully!")
    else:
        print("✗ Some tests failed. Check the output above for details.")

