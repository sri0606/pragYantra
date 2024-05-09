from transformers import T5Tokenizer, T5ForConditionalGeneration

class Summarizer:
    """
    Summarizer class
    """

    def __init__(self, model="t5-base") -> None:
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.tokenizer = T5Tokenizer.from_pretrained(model)

    def tokenize_and_truncate(self, text, max_length):
        # Tokenize input text 
        inputs = self.tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
        return inputs

    def chunk_text(self, text, max_length):

        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        return chunks

    def summarize(self, transcript_text, max_length=150, min_length=50, length_penalty=15, num_beams=4, max_input_length=512):
        """
        Summarize the given transcript.

        Parameters:
        @transcript_text: The input transcript text to be summarized.
        @max_length : The maximum length of the generated summary.
        @min_length : The minimum length of the generated summary.
        @length_penalty : Length penalty to encourage or discourage longer summaries.
        @num_beams : Number of beams to use in beam search during generation.
        @max_input_length: Maximum length for input text chunks.

        Returns:
        - str: The generated summary of the input transcript.
        """
        # Chunk the input text
        text_chunks = self.chunk_text(transcript_text, max_input_length)
       
        # Summarize each chunk
        summaries = []
        for chunk in text_chunks:
            chunk = "summarize: "+chunk
            # Tokenize and truncate input chunk
            inputs = self.tokenize_and_truncate(chunk, max_input_length)

            # Generate summary for the chunk
            summary_ids = self.model.generate(**inputs, max_length=max_length, min_length=min_length, length_penalty=length_penalty, num_beams=num_beams)

            summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary_text)

        # merge all summaries
        final_summary = " ".join(summaries)
        return final_summary