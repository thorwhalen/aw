"""
Test suite demonstrating FileStore auto-initialization functionality.

This test suite verifies that the enhanced FileStore can:
1. Create missing files with initial content
2. Create missing key_path sections
3. Preserve existing content when adding new sections
"""

import tempfile
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'i' / 'config2py'))

from config2py.sync_store import FileStore


def test_missing_file():
    """Test that missing files are created with factory content."""
    print("Test 1: Missing file creation")
    print("-" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'new_config.json'
        
        # Verify file doesn't exist
        assert not config_path.exists(), "File should not exist initially"
        print(f"✓ File doesn't exist: {config_path}")
        
        # Create FileStore with factories
        store = FileStore(
            config_path,
            key_path='section',
            create_file_content=lambda: {},
            create_key_path_content=lambda: {}
        )
        
        # Verify file was created
        assert config_path.exists(), "File should be created"
        print(f"✓ File created automatically")
        
        # Verify content structure
        with open(config_path) as f:
            content = json.load(f)
        
        assert 'section' in content, "section key should exist"
        assert content['section'] == {}, "section should be empty dict"
        print(f"✓ Correct structure: {content}")
        
        # Add data
        store['item'] = {'value': 123}
        
        # Verify persistence
        with open(config_path) as f:
            content = json.load(f)
        
        assert content['section']['item']['value'] == 123
        print(f"✓ Data persisted correctly: {content}")
        
    print("✅ Test 1 passed\n")


def test_missing_key_path():
    """Test that missing key_path sections are created."""
    print("Test 2: Missing key_path creation")
    print("-" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.json'
        
        # Create file with some existing content
        initial_content = {
            'existing_section': {'key': 'value'},
            'other_data': [1, 2, 3]
        }
        
        with open(config_path, 'w') as f:
            json.dump(initial_content, f)
        
        print(f"✓ Created file with initial content: {initial_content}")
        
        # Open with key_path that doesn't exist
        store = FileStore(
            config_path,
            key_path='new_section',
            create_key_path_content=lambda: {}
        )
        
        # Verify key_path was created
        with open(config_path) as f:
            content = json.load(f)
        
        assert 'new_section' in content, "new_section should exist"
        assert content['new_section'] == {}, "new_section should be empty"
        
        # Verify existing content preserved
        assert content['existing_section'] == initial_content['existing_section']
        assert content['other_data'] == initial_content['other_data']
        print(f"✓ Key path created, existing content preserved")
        
        # Add data to new section
        store['item'] = 'test'
        
        # Verify everything is correct
        with open(config_path) as f:
            content = json.load(f)
        
        assert content['new_section']['item'] == 'test'
        assert content['existing_section']['key'] == 'value'
        print(f"✓ All content correct: {content}")
        
    print("✅ Test 2 passed\n")


def test_claude_desktop_config():
    """Test the actual claude_desktop_config use case."""
    print("Test 3: Claude Desktop config simulation")
    print("-" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Import inside test to use temp directory
        sys.path.insert(0, str(Path(__file__).parent))
        from aw.util import claude_desktop_config
        
        config_path = Path(tmpdir) / 'claude_desktop_config.json'
        
        print(f"✓ Config path: {config_path}")
        print(f"✓ File exists before: {config_path.exists()}")
        
        # This is what the agent does
        mcp = claude_desktop_config(config_dir=tmpdir)
        
        print(f"✓ File exists after: {config_path.exists()}")
        
        # Add server config
        mcp['download'] = {
            'command': 'python',
            'args': ['/path/to/agent.py']
        }
        
        print(f"✓ Added download server")
        
        # Verify file structure
        with open(config_path) as f:
            content = json.load(f)
        
        expected_structure = {
            'mcpServers': {
                'download': {
                    'command': 'python',
                    'args': ['/path/to/agent.py']
                }
            }
        }
        
        assert content == expected_structure, f"Content mismatch: {content}"
        print(f"✓ File structure correct")
        print(f"  {json.dumps(content, indent=2)}")
        
    print("✅ Test 3 passed\n")


def test_without_factories_fails():
    """Test that without factories, missing content raises errors."""
    print("Test 4: Default behavior (no factories)")
    print("-" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'missing.json'
        
        print(f"✓ Testing with missing file, no factory")
        
        try:
            # Should raise FileNotFoundError
            store = FileStore(config_path, key_path='section')
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            print(f"✓ Correctly raised FileNotFoundError: {e}")
        
        # Create file but missing key_path
        with open(config_path, 'w') as f:
            json.dump({'other': 'data'}, f)
        
        print(f"✓ Testing with existing file, missing key_path, no factory")
        
        try:
            # Should raise KeyError
            store = FileStore(config_path, key_path='missing_section')
            assert False, "Should have raised KeyError"
        except KeyError as e:
            print(f"✓ Correctly raised KeyError: {e}")
    
    print("✅ Test 4 passed\n")


if __name__ == '__main__':
    print("=" * 70)
    print("FileStore Auto-Initialization Test Suite")
    print("=" * 70)
    print()
    
    test_missing_file()
    test_missing_key_path()
    test_claude_desktop_config()
    test_without_factories_fails()
    
    print("=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
