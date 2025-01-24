def get_base_css() -> str:
    return """
    @media (prefers-color-scheme: light) {
        :root {
            --bg-human: #f0f0f0;
            --bg-thought: #e6f3ff;
            --bg-code: #f8f8f8;
            --bg-assistant: #e8f5e9;
            --bg-execution: #fff3e0;
            --text-normal: #000000;
            --text-muted: #666666;
            --border-color: #dddddd;
            --link-color: #0366d6;
        }
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --bg-human: #2a2a2a;
            --bg-thought: #1a2634;
            --bg-code: #1e1e1e;
            --bg-assistant: #1b2b1e;
            --bg-execution: #2b2317;
            --text-normal: #e0e0e0;
            --text-muted: #a0a0a0;
            --border-color: #404040;
            --link-color: #58a6ff;
        }
    }

    .event {
        margin: 8px 0;
        padding: 12px;
        border-radius: 6px;
        color: var(--text-normal);
        line-height: 1.5;
    }

    .event a {
        color: var(--link-color);
        text-decoration: none;
    }

    .event a:hover {
        text-decoration: underline;
    }

    .human {
        background: var(--bg-human);
    }

    .assistant-thought {
        background: var(--bg-thought);
        font-style: italic;
        color: var(--text-muted);
    }

    .code {
        background: var(--bg-code);
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    }

    .assistant {
        background: var(--bg-assistant);
    }

    .execution {
        background: var(--bg-execution);
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        white-space: pre-wrap;
        word-wrap: break-word;
    }

    .session {
        border: 1px solid var(--border-color);
        padding: 16px;
        margin: 16px 0;
        border-radius: 8px;
    }

    /* Markdown content styling */
    .event h1, .event h2, .event h3, .event h4, .event h5, .event h6 {
        margin-top: 24px;
        margin-bottom: 16px;
        font-weight: 600;
        line-height: 1.25;
    }

    .event p {
        margin-bottom: 16px;
    }

    .event ul, .event ol {
        margin-bottom: 16px;
        padding-left: 2em;
    }

    .event table {
        border-collapse: collapse;
        margin: 16px 0;
        width: 100%;
    }

    .event th, .event td {
        padding: 6px 13px;
        border: 1px solid var(--border-color);
    }

    .event tr {
        background-color: var(--bg-human);
    }
    """
