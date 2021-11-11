
import torch

def train(model, train_loader, optimizer, device, logger):
        model.train()
        loss_sum = 0
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs[0]
            loss.backward()
            optimizer.step()

            if (iter + 1) % 100 == 0:
                logger.info(' step : {}/{} Loss: {:.4f}'.format(
                    iter, 
                    str(len(train_loader)),
                    loss.detach())
                )
            


def valid(model, dev_loader, device, tokenizer, logger):

    model.eval()
    pred_texts = []
    ans_texts = []
    loss_sum = 0
    with torch.no_grad():
        for iter,batch in enumerate(dev_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)

            pred_start_positions = torch.argmax(outputs['start_logits'], dim=1).to('cpu')
            pred_end_positions = torch.argmax(outputs['end_logits'], dim=1).to('cpu')
            for b in range(len(pred_start_positions)):
                ans_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[b][start_positions[b]:end_positions[b]+1]))
                pred_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[b][pred_start_positions[b]:pred_end_positions[b]+1]))
                ans_texts.append(ans_text)
                pred_texts.append(pred_text)
            loss = outputs[0].to('cpu')
            loss_sum += loss
    
    return pred_texts, ans_texts, loss_sum/iter
        
        

            