        context = (
            f"{kohana_personality}\n"
            f"Current date and time: {now_str}.\n"
            f"You are speaking to: {user_profile['name']} ({user_profile['relationship']}). "
            f"{thought_annotation}"  # only present if use_reasoning=True
            f"Relevant memories: {memories_text or 'none'}.\n"
            f"Stay concise, and only respond as the Kitsune girl, Yuki Kohana.\n"
            f"Do not include physical or emotional reaction tags in your response.\n"
            f"Conversation History: {history}\n"
            f"{user_profile['name']}: {prompt}\nKohana:"
        )