# Multi-Query Web Search Application

An AI-augmented search application that expands your search query and explores multiple perspectives of your original question.

## Features

- **Query Expansion**: Uses an OpenAI-compatible LLM to generate 5 diverse, related search queries
- **Parallel Search**: Performs multiple web searches using the DuckDuckGo search engine
- **Result Deduplication**: Removes duplicate results based on URLs
- **AI Re-ranking**: Re-ranks the combined search results based on relevance to your original query
- **Clean UI**: Simple, responsive user interface

## How It Works

1. You enter a search query in the web interface
2. The backend generates 5 diverse, related search queries using an LLM
3. The application performs 5 parallel web searches using DuckDuckGo
4. Results are deduplicated and then re-ranked by the LLM based on relevance to your original query
5. The final, re-ranked results are displayed in the web interface

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- An OpenAI API key (or compatible API key for a local LLM)

### Installation

1. Clone this repository or download the source code:

```bash
git clone https://github.com/yourusername/multi-query-search.git
cd multi-query-search
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

- **Windows**:
  ```bash
  venv\Scripts\activate
  ```

- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

5. Create a `.env` file based on the `.env.example` file:

```bash
cp .env.example .env
```

6. Edit the `.env` file and add your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

### Running the Application

1. Start the Flask server:

```bash
python app.py
```

2. Open your web browser and navigate to:

```
http://localhost:5000
```

## Using a Local LLM

To use a local LLM instead of OpenAI's API:

1. Set up your local LLM with an OpenAI-compatible API (e.g., LM Studio, LocalAI, Ollama, etc.)
2. In your `.env` file, add the base URL of your local LLM API:

```
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_MODEL=your_local_model_name
```

## Customization

- **Search Results**: You can adjust the number of results per query in the `search_duckduckgo` function in `app.py`
- **Model Selection**: Change the LLM model by setting the `OPENAI_MODEL` environment variable
- **UI Styling**: Modify the CSS in `templates/index.html` to change the appearance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Uses the [DuckDuckGo Search Python library](https://github.com/deedy5/duckduckgo_search)
- Powered by [OpenAI API](https://openai.com/api/) or compatible LLM providers 