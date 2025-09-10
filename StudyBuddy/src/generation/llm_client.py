import google.generativeai as genai
from typing import Dict, Any, Optional, List
from config.settings import settings

class GeminiClient:
    def __init__(self):
        settings.validate_required_keys()
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        self.generation_config = genai.GenerationConfig(
            temperature=settings.TEMPERATURE,
            max_output_tokens=settings.MAX_TOKENS,
        )
    
    def generate_response(
        self, 
        query: str, 
        context: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(query, context, conversation_history)
        
        try:
            response = self.model.generate_content(
                f"{system_prompt}\n\n{user_prompt}",
                generation_config=self.generation_config
            )
            
            return {
                'response': response.text,
                'success': True,
                'error': None,
                'usage': {
                    'prompt_tokens': response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else None,
                    'completion_tokens': response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else None,
                    'total_tokens': response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else None,
                }
            }
        except Exception as e:
            return {
                'response': f"I apologize, but I encountered an error while processing your request: {str(e)}",
                'success': False,
                'error': str(e),
                'usage': None
            }
    
    def _create_system_prompt(self) -> str:
        return """You are a helpful AI assistant that answers questions based on the user's personal knowledge base. 

Your role:
- Answer questions using ONLY the provided context from the user's documents
- Be accurate, helpful, and concise
- If the context doesn't contain enough information to answer the question, clearly state this
- Always cite your sources when possible (mention the document name)
- Maintain a friendly and professional tone

Guidelines:
- DO NOT make up information not present in the context
- If asked about something not in the context, suggest the user might need to add more relevant documents
- When referencing information, mention which document it came from
- Keep responses focused and relevant to the question asked"""
    
    def _create_user_prompt(
        self, 
        query: str, 
        context: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        prompt_parts = []
        
        # Add conversation history if provided
        if conversation_history:
            prompt_parts.append("Previous conversation:")
            for turn in conversation_history[-3:]:  # Last 3 turns
                prompt_parts.append(f"User: {turn.get('user', '')}")
                prompt_parts.append(f"Assistant: {turn.get('assistant', '')}")
            prompt_parts.append("")
        
        # Add context
        prompt_parts.append("Context from your knowledge base:")
        prompt_parts.append(context)
        prompt_parts.append("")
        
        # Add current question
        prompt_parts.append(f"User question: {query}")
        prompt_parts.append("")
        prompt_parts.append("Please answer based on the provided context:")
        
        return "\n".join(prompt_parts)
    
    def generate_simple_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"