import gradio as gr
import openai
import requests
import os
import json
from typing import List, Tuple, Dict
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader


def scrape_website_links(url: str) -> List[str]:
    """Scrapes all hyperlinks from a website."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]
        return list(set(links))  # Remove duplicates
    except requests.RequestException as e:
        return [f"Error: {str(e)}"]


def scrape_page_content(url: str) -> str:
    """Scrapes visible text content from a webpage."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join([p.get_text(strip=True) for p in soup.find_all("p")])
        return text.strip()
    except requests.RequestException as e:
        return f"Error scraping {url}: {str(e)}"


def query_openai(system_prompt: str, user_prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """Queries OpenAI API with prompts."""
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error querying OpenAI: {str(e)}"


def summarize_content(content: str, chunk_size: int = 3000) -> str:
    """Summarizes large content by chunking and querying OpenAI."""
    chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
    summaries = []
    for chunk in chunks:
        system_prompt = "Summarize the following text:"
        summary = query_openai(system_prompt, chunk)
        summaries.append(summary)
    return " ".join(summaries)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

import json

import json

def collect_relevant_links(system_prompt: str, company_url: str, position: str) -> List[str]:
    """
    Filters provided links based on a system prompt using the OpenAI API.

    Args:
        system_prompt (str): The system-level instruction to guide filtering.
        company_url (str): The URL of the company website to scrape for links.
        position (str): The job position to tailor the filtering criteria.

    Returns:
        List[str]: A list of relevant links based on the filtering criteria.
    """
    # Step 1: Scrape all links from the given company URL
    all_links = scrape_website_links(company_url)
    if not all_links:
        return ["Error: No links found on the website."]

    # Step 2: First round of filtering with OpenAI API
    user_prompt = (
        f"Filter the following links to identify those relevant for a {position} position. "
        f"Focus on pages about the company, its research, products, services, or careers. "
        f"Exclude unrelated pages like Terms of Service or Privacy Policies.\n\nLinks: {all_links}"
    )

    try:
        first_response = query_openai(system_prompt, user_prompt)
        # print(f"First response: {first_response}")
        
        # Attempt to parse the response as JSON
        try:
            parsed_response = json.loads(first_response)
            relevant_links = parsed_response.get("relevant_links", [])
            print(f"Initial relevant links count: {len(relevant_links)}")
        except json.JSONDecodeError:
            return ["Error: Failed to parse OpenAI response as JSON."]
    except Exception as e:
        return [f"Error: Failed to query OpenAI - {str(e)}"]

    if not relevant_links:
        return ["Error: No relevant links found in the first filtering step."]

    # Step 3: Scrape further links from initially filtered links
    additional_links = []
    for link_obj in relevant_links:
        additional_links.extend(scrape_website_links(link_obj["link"]))

    # Combine the original filtered links with additional scraped links
    combined_links = list(set([link["link"] for link in relevant_links] + additional_links))
    # print(f"Total links after further scraping: {len(combined_links)}")

    # Step 4: Second round of filtering with OpenAI API
    second_user_prompt = (
        f"Further refine the following links to identify the most relevant ones for a {position} position. "
        f"Ensure these pages provide substantial information about the company, its mission, values, products, or career opportunities.\n\nLinks: {combined_links}"
    )

    try:
        second_response = query_openai(system_prompt, second_user_prompt)
        print(f"Second response (raw): {second_response}")  # Print raw second response

        # Attempt to parse the second response as JSON
        try:
            parsed_second_response = json.loads(second_response)
            final_relevant_links = parsed_second_response.get("relevant_links", [])
            print(f"Final relevant links count: {len(final_relevant_links)}")

            # Sort the final links by priority
            sorted_links = sorted(final_relevant_links, key=lambda x: x["priority"])
            return [link["link"] for link in sorted_links]
        except json.JSONDecodeError as e:
            print(f"Error parsing second response as JSON: {e}")
            return ["Error: Failed to parse second OpenAI response as JSON."]
    except Exception as e:
        return [f"Error: Failed to query OpenAI in the second step - {str(e)}"]



def analyze_company_resume(company_url: str, position: str, resume_path: str) -> str:
    """Analyzes a company's website and a resume to generate a summary of relevance."""
    # Scrape and filter links from the company's website
 
    # all_links = scrape_website_links(company_url)
    # relevant_links = [link for link in all_links if any(keyword in link for keyword in ["about", "research", "products", "careers"])]
    # re
    link_system_prompt = """You are an expert assistant tasked with identifying and prioritizing relevant web pages to research a company's online presence and offerings. Given a list of links from the company's website:

            1. Filter the links that are most relevant to understanding the company. Focus on pages that include:
            - Information about the company (e.g., 'About Us', 'Mission', 'Leadership').
            - Research efforts or innovations.
            - Products, services, or solutions offered.
            - Career opportunities (e.g., 'Careers', 'Jobs').

            2. Exclude links that lead to irrelevant or non-informative pages such as:
            - Terms of Service, Privacy Policies, Cookie Policies.
            - Generic support pages or contact forms.
            - Pages unrelated to the company (e.g., external links, media policies).

            3. Prioritize the relevant links based on their importance:
            - Assign priority 1 to the most critical page for understanding the company.
            - Increment the priority for less critical but still relevant pages (e.g., priority 2, 3, etc.).

            4. Return the response in the following JSON format:
                {
                    "relevant_links": [
                        {"link": "link1", "priority": 1},
                        {"link": "link2", "priority": 2},
                        {"link": "link3", "priority": 3}
                    ],
                    "excluded_links": ["link4", "link5", "link6"]
                }
        """
# Collect and summarize content from relevant links
    relevant_links = collect_relevant_links(link_system_prompt,company_url,position)
    
    context = ""
    for link in relevant_links[:20]:  # Limit to the first 10 links for efficiency
        page_content = scrape_page_content(link)
        context += f"\n\n{page_content}"

    # Summarize the context to fit within the API token limit
    summarized_context = summarize_content(context)

    # Extract and prepare the resume text
    resume_text = extract_text_from_pdf(resume_path)

    # Query OpenAI to analyze the company and resume
    system_prompt = (
        """You are a helpful and insightful assistant. Given the company context and the candidate's resume, provide a detailed analysis of how the candidate's experience and skills align with the company's mission, research, products, and values. Specifically, consider the following:

        Company Context: Summarize the company's mission, core values, and key products or research initiatives.
        Candidate Experience: Identify the candidate's relevant skills, past experiences, and accomplishments.
        Alignment:
        Explain how the candidate's experience directly aligns with the company's products, research efforts, and innovation.
        Discuss how the candidate's skills and values complement the company's mission and culture.
        Fit for the Role: Provide insights on why the candidate would be a strong fit for the company and this position, considering both technical and cultural aspects.
        Make sure to provide a well-rounded explanation that clearly ties the candidate's background with the company's goals and initiatives."""
    )
    user_prompt = f"""I have provided my resume and information about a company's website. \
    Could you analyze both and provide a concise summary of the company's mission, \
    values, products, research, and ongoing projects? Additionally, please assess my \
    resume and explain how my skills, experience, and qualifications align with \
    opportunities at this company. Keep the response brief, focused, and insightful.
    Resume: {resume_text}\n\nContext: \n{summarized_context}."""
    # user_prompt = f"Resume:\n{resume_text}\n\nCompany Context:\n{summarized_context}"
    analysis = query_openai(system_prompt, user_prompt)

    return analysis


def start_gradio_chat():
    
    """Launches a Gradio interface for the analysis tool."""
    def process_inputs(company_url: str, position: str, resume_file) -> str:
        if resume_file:
            resume_path = resume_file.name
            return analyze_company_resume(company_url, position, resume_path)
        else:
            return "Please upload a valid resume file."

    iface = gr.Interface(
        fn=process_inputs,
        inputs=[
            gr.Textbox(label="Company URL"),
            gr.Textbox(label="Position"),
            gr.File(label="Upload Resume (PDF)", file_types=[".pdf"]),
        ],
        outputs="text",
        title="Company Analysis Tool",
    )

    iface.launch()


if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY", "api key")
    start_gradio_chat()
