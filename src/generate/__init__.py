"""
Text generation from LLM.

This module, and its submodules, should be concerned with taking in a ChatML conversation and generating a _meaningful_ assistant response.

This means we should:
- Use constrained generation (`constrain` module) to enforce a certain response structure, even without fine-tuning.
- Be mindful of the system prompt / steering - very important so that the model understands
"""
