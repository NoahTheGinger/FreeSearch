import os
import json
import re
import logging
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from openai import OpenAI
from duckduckgo_search import DDGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE", None)  # Optional custom base URL
)

# Default model if not specified in environment
DEFAULT_MODEL = "gpt-4o-mini"

def extract_json_from_llm_response(response_text):
    """
    Extract and validate JSON from an LLM response.
    Handles cases where the model might add extra text or code fences.
    """
    logger.info("Extracting JSON from LLM response")
    
    # Try to find JSON array in the response using regex
    # This handles cases where the model wraps JSON in ```json or similar
    json_match = re.search(r'(\[.+\]|\{.+\})', response_text, re.DOTALL)
    if json_match:
        try:
            potential_json = json_match.group(1)
            parsed_json = json.loads(potential_json)
            return parsed_json
        except json.JSONDecodeError:
            logger.warning(f"Found JSON-like content but couldn't parse: {potential_json}")
    
    # Direct parse attempt
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse direct JSON from response: {response_text}")
    
    # Progressive string cleaning as a fallback
    cleaned_response = response_text.strip()
    
    # Remove markdown code blocks if present
    if cleaned_response.startswith("```") and "```" in cleaned_response[3:]:
        cleaned_response = cleaned_response.split("```", 2)[1]
        if cleaned_response.startswith("json"):
            cleaned_response = cleaned_response[4:].strip()
    
    # Final attempt with cleaned response
    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        logger.error(f"All JSON extraction attempts failed for response: {response_text}")
        raise ValueError("Could not parse valid JSON from LLM response")

def generate_queries(original_query):
    """
    Call LLM to generate 5 diverse search queries based on the original query.
    """
    logger.info(f"Generating diverse queries for: {original_query}")
    
    try:
        # Create prompt for query expansion
        prompt = f"Given the user query: {original_query}, generate 5 distinct, related search queries that explore different aspects or synonyms. Return ONLY a JSON array of strings, nothing else. Do not include any triple backticks or extra text."
        
        # Call OpenAI API
        model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful search query expansion assistant. You generate diverse, related search queries to explore different aspects of a topic. Return ONLY JSON without any additional text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=400
        )
        
        # Extract response content
        response_text = response.choices[0].message.content
        logger.debug(f"Raw LLM response for query expansion: {response_text}")
        
        # Extract and validate JSON
        expanded_queries = extract_json_from_llm_response(response_text)
        
        # Validate that we got a list of strings
        if not isinstance(expanded_queries, list):
            logger.warning(f"Expected list but got {type(expanded_queries)}")
            raise ValueError("LLM did not return a valid list of queries")
        
        # Ensure we have at least one query
        if not expanded_queries:
            logger.warning("LLM returned empty list of queries")
            expanded_queries = [original_query]
        
        # Ensure all items are strings
        expanded_queries = [str(q) for q in expanded_queries]
        
        logger.info(f"Generated {len(expanded_queries)} expanded queries")
        return expanded_queries
        
    except Exception as e:
        logger.error(f"Error generating expanded queries: {str(e)}")
        # Fall back to original query on error
        return [original_query]

def search_duckduckgo(queries, max_results_per_query=5):
    """
    Perform searches on DuckDuckGo for each query.
    Returns aggregated, deduplicated results.
    """
    logger.info(f"Searching DuckDuckGo for {len(queries)} queries")
    
    all_results = []
    seen_urls = set()
    
    with DDGS() as ddgs:
        for query in queries:
            try:
                logger.info(f"Searching for query: {query}")
                results = list(ddgs.text(query, max_results=max_results_per_query))
                logger.info(f"Found {len(results)} results for query: {query}")
                
                # Add unique results to the aggregated list
                for result in results:
                    url = result.get('href', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(result)
                
            except Exception as e:
                logger.error(f"Error searching DuckDuckGo for query '{query}': {str(e)}")
    
    logger.info(f"Total unique results after deduplication: {len(all_results)}")
    return all_results

def rerank_results(original_query, results):
    """
    Call LLM to re-rank search results based on relevance to the original query.
    """
    if not results:
        logger.warning("No results to re-rank")
        return []
    
    logger.info(f"Re-ranking {len(results)} search results")
    
    try:
        # Prepare results for LLM (simplified to reduce token usage)
        simplified_results = []
        for i, result in enumerate(results):
            simplified_results.append({
                "id": i,
                "title": result.get("title", ""),
                "href": result.get("href", ""),
                "body": result.get("body", "")
            })
        
        # Create prompt for re-ranking
        prompt = f"""Given the original user query: "{original_query}", and the following search results:
{json.dumps(simplified_results, indent=2)}

Re-rank the results by likely relevance to the original query. DO NOT remove any results. Return ONLY a JSON array of integers representing the new order of the results (indices starting from 0). Do not include any triple backticks or extra text."""
        
        # Call OpenAI API
        model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful search result ranking assistant. You determine the most relevant order of search results for a user's query without filtering any results. Return ONLY JSON without any additional text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400
        )
        
        # Extract response content
        response_text = response.choices[0].message.content
        logger.debug(f"Raw LLM response for result ranking: {response_text}")
        
        # Extract and validate JSON
        rankings = extract_json_from_llm_response(response_text)
        
        # Validate that we got a list of integers
        if not isinstance(rankings, list):
            logger.warning(f"Expected list but got {type(rankings)}")
            raise ValueError("LLM did not return a valid list of indices")
        
        # Validate indices
        valid_indices = []
        for idx in rankings:
            try:
                idx = int(idx)
                if 0 <= idx < len(results):
                    valid_indices.append(idx)
                else:
                    logger.warning(f"Index {idx} out of range, ignoring")
            except (ValueError, TypeError):
                logger.warning(f"Invalid index value: {idx}, ignoring")
        
        # If we have no valid indices, return original order
        if not valid_indices:
            logger.warning("No valid indices in ranking, returning original order")
            return results
        
        # Add any missing indices in original order
        existing_indices = set(valid_indices)
        for i in range(len(results)):
            if i not in existing_indices:
                valid_indices.append(i)
        
        # Reorder results
        reranked_results = [results[idx] for idx in valid_indices]
        logger.info("Results successfully re-ranked")
        return reranked_results
        
    except Exception as e:
        logger.error(f"Error re-ranking results: {str(e)}")
        # Fall back to original results order on error
        return results

@app.route('/')
def index():
    """Render the main search page."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests."""
    try:
        # Get query from request
        data = request.json
        if not data or 'query' not in data:
            logger.warning("Invalid request: missing query")
            return jsonify({"error": "Missing query parameter"}), 400
        
        original_query = data['query'].strip()
        if not original_query:
            logger.warning("Empty query received")
            return jsonify({"error": "Query cannot be empty"}), 400
        
        logger.info(f"Received search request for query: {original_query}")
        
        # Generate expanded queries
        expanded_queries = generate_queries(original_query)
        
        # Perform DuckDuckGo searches
        search_results = search_duckduckgo(expanded_queries)
        
        # Re-rank results if we have any
        if search_results:
            reranked_results = rerank_results(original_query, search_results)
        else:
            reranked_results = []
        
        # Return results
        return jsonify({
            "query": original_query,
            "expanded_queries": expanded_queries,
            "results": reranked_results
        })
    
    except Exception as e:
        logger.error(f"Error handling search request: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # Check for required environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is not set")
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it in your .env file or environment.")
        exit(1)
    
    # Log startup information
    model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    logger.info(f"Starting Multi-Query Search with model: {model}")
    
    # Run the Flask app
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1", host="0.0.0.0", port=int(os.environ.get("PORT", 5000))) 