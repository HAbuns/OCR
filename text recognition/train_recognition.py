# DATASET
class STRDataset(Dataset):
    def __init__(self, X, y, char_to_idx, max_label_len, label_encoder=None, transform=None):
        self.img_paths = X
        self.labels = y
        self.char_to_idx = char_to_idx
        self.max_label_len = max_label_len
        self.label_encoder = label_encoder
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        if self.label_encoder:
            encoded_label, label_len = self.label_encoder(label, self.char_to_idx, self.max_label_len)

        return img, encoded_label, label_len
    
# Model - CRNN
class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers, dropout=0.2, unfreeze_layers=3):
        super(CRNN, self).__init__()

        backbone = timm.create_model('resnet101', in_chans=1, pretrained=True)
        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((1, None)))
        self.backbone = nn.Sequential(*modules)

        # Unfreeze n layers cuối cùng
        for param in self.backbone[-unfreeze_layers:].parameters():
            param.requires_grad = True

        self.mapSeq = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, vocab_size),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, x):
        x = self.backbone(x)                   # [B, C, H=1, W]
        x = x.permute(0, 3, 1, 2)              # [B, W, C, H]
        x = x.view(x.size(0), x.size(1), -1)   # [B, W, C*H]
        x = self.mapSeq(x)                     # [B, W, 512]
        x, _ = self.lstm(x)                    # [B, W, hidden*2]
        x = self.layer_norm(x)
        x = self.out(x)                        # [B, W, vocab_size]
        return x.permute(1, 0, 2)              # [W, B, vocab_size] for CTC Loss
    
#TRAINING LOOP
def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for inputs, labels, label_lens in train_loader:
            inputs, labels, label_lens = inputs.to(device), labels.to(device), label_lens.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            logit_lens = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long).to(device)

            loss = criterion(outputs, labels, logit_lens, label_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            batch_losses.append(loss.item())

        train_loss = sum(batch_losses) / len(batch_losses)
        val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

#VALIDATION
def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for inputs, labels, label_lens in dataloader:
            inputs, labels, label_lens = inputs.to(device), labels.to(device), label_lens.to(device)
            outputs = model(inputs)
            logit_lens = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long).to(device)
            loss = criterion(outputs, labels, logit_lens, label_lens)
            losses.append(loss.item())

    return sum(losses) / len(losses)

if __name__ == "__main__":
    train_dataset = STRDataset(X_train, y_train, char_to_idx, max_label_len, label_encoder=encode, transform=data_transform['train'])
    val_dataset = STRDataset(X_val, y_val, char_to_idx, max_label_len, label_encoder=encode, transform=data_transform['val'])
    test_dataset = STRDataset(X_test, y_test, char_to_idx, max_label_len, label_encoder=encode, transform=data_transform['val'])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CRNN(vocab_size, hidden_size=256, n_layers=3, dropout=0.2, unfreeze_layers=3).to(device)

    criterion = nn.CTCLoss(blank=char_to_idx['-'], zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(100 * 0.4), gamma=0.1)

    train_losses, val_losses = fit(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=100)
