from use_embed import USEEmbedder


lang_embed_model = USEEmbedder()
print("USE v5 loaded")

sentence = "The quick brown fox jumped over the lazy dog"
embedding = lang_embed_model.embed(sentence)
print(len(embedding), "dimensional embedding vector:")
print("[", embedding[0], embedding[1], embedding[2], " ... ", embedding[-3], embedding[-2], embedding[-1],"]")
