# Add to scripts/test_integration.py
def test_multiview_consistency():
    # Load model
    model = MultiViewDiffusion(...)
    
    # Load test data
    test_depth1 = torch.randn(1, 1, 256, 256)
    test_depth2 = torch.randn(1, 1, 256, 256)
    test_pose = torch.eye(4).unsqueeze(0)
    
    # Compute consistency loss
    loss = model._compute_consistency_loss(
        torch.stack([test_depth1, test_depth2]), 
        torch.stack([test_pose, test_pose])
    )
    
    print(f"Consistency loss: {loss.item()}")