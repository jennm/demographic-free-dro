import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, input_dim, cdim=32, hdim=64):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, cdim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(cdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Custom cosine similarity loss function
class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, x, y):
        cos = nn.CosineSimilarity(dim=1)
        return 1 - cos(x, y).mean()  # We want to minimize the cosine similarity, so we use 1 - similarity
    
    
def train_encoder_decoder(model, dataloader, criterion, optimizer, num_epochs=15):
    for epoch in range(num_epochs):
        for batch in dataloader:
            original_batch, projected_batch = batch
            optimizer.zero_grad()
            outputs = model(original_batch)
            loss = criterion(outputs, projected_batch)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model