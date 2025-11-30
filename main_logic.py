# SAFE DEMO VERSION ONLY
# (No RAG, No embeddings, No multi-tenant, No DB, No retrieval)

def simple_generate_email(purpose, tone, key_points):
    """
    This is a VERY simplified version of the real engine.
    """

    template = f"""
Subject: Regarding {purpose}

Hi,

I hope you are doing well.

Here is a brief follow-up regarding: {purpose}

Key Points:
{key_points}

Tone Used: {tone}

Best regards,
AI Email Assistant Demo
"""
    return template.strip()
