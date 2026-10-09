# Contributing to FinanceRAG

Thank you for your interest in contributing to FinanceRAG! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/financial-rag-system.git`
3. Create a virtual environment: `python -m venv venv`
4. Install dependencies: `pip install -r requirements.txt`
5. Install dev dependencies: `pip install pytest black flake8 mypy`

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write clear, readable code
- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Include type hints where appropriate

### 3. Test Your Changes

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src tests/

# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/
```

### 4. Commit Your Changes

Use clear, descriptive commit messages:

```bash
git add .
git commit -m "Add: New feature for document comparison"
# or
git commit -m "Fix: Chunking overlap calculation error"
# or
git commit -m "Docs: Update API documentation"
```

Commit message prefixes:
- `Add:` - New features
- `Fix:` - Bug fixes
- `Update:` - Updates to existing features
- `Docs:` - Documentation changes
- `Test:` - Test additions or changes
- `Refactor:` - Code refactoring
- `Style:` - Code style changes

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title describing the change
- Description of what was changed and why
- Reference to any related issues
- Screenshots if UI changes

## Code Style

### Python

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use descriptive variable names
- Add docstrings to all public functions

Example:

```python
def process_document(file_path: str, chunk_size: int = 512) -> Dict[str, Any]:
    """
    Process a document and create chunks.
    
    Args:
        file_path: Path to the document file
        chunk_size: Target size for chunks in tokens
        
    Returns:
        Dict containing text, metadata, and chunks
        
    Raises:
        ValueError: If file type is not supported
    """
    pass
```

### JavaScript

- Use ES6+ syntax
- Use meaningful variable names
- Add JSDoc comments for functions
- Use const/let, not var

## Testing

- Write tests for new features
- Maintain >80% code coverage
- Test both happy paths and error cases
- Use descriptive test names

Example:

```python
def test_chunk_text_preserves_sentences():
    """Test that chunking preserves sentence boundaries"""
    processor = DocumentProcessor(chunk_size=100)
    text = "First sentence. Second sentence."
    chunks = processor.chunk_text(text)
    
    for chunk in chunks:
        assert chunk['text'][-1] in '.!?'
```

## Documentation

- Update README.md if adding new features
- Add docstrings to all functions
- Update API documentation in docs/API.md
- Add examples for new functionality

## Pull Request Guidelines

Your PR should:
- Have a clear, descriptive title
- Include a detailed description of changes
- Reference any related issues
- Pass all tests
- Maintain or improve code coverage
- Follow the code style guidelines
- Include documentation updates if needed

## Feature Requests and Bug Reports

### Bug Reports

Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Relevant logs or error messages

### Feature Requests

Include:
- Clear description of the feature
- Use case and rationale
- Proposed implementation (optional)
- Examples of similar features in other projects

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

## Questions?

- Open an issue for questions
- Join discussions in GitHub Discussions
- Check existing issues and documentation first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to FinanceRAG! 🎉
