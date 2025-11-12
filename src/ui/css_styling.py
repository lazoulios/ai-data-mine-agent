INLINE_CSS_STYLING = """
<style>
.main .block-container {
    padding-top: 2rem;
    max-width: 1000px;
}

.stTextArea textarea {
    height: 100px;
}

.loading-dots {
    display: inline-block;
    animation: loading-dots 1.5s infinite;
}

@keyframes loading-dots {
    0% { content: '.'; }
    33% { content: '..'; }
    66% { content: '...'; }
    100% { content: '.'; }
}

/* Tool calls button styling */
button[data-testid="baseButton-secondary"] {
    border-radius: 50%;
    width: 35px;
    height: 35px;
    padding: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}

/* Tool calls expandable section styling */
.stExpander {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    margin-bottom: 10px;
}

.stExpander {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    margin-bottom: 10px;
}
</style>
"""