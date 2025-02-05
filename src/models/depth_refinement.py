# src/models/depth_refinement.py
class DepthRefinementPipeline:
    def __init__(self, diffusion_model, feature_extractor):
        self.diffusion = diffusion_model
        self.feature_extractor = feature_extractor
        
    def refine_depth_sequence(self, depth_frames, poses):
        """Process a sequence of depth frames with temporal consistency"""
        refined_depths = []
        features = []
        
        for frame, pose in zip(depth_frames, poses):
            # Refine single frame
            refined = self.diffusion.denoise(frame)
            
            # Extract geometric features
            feat = self.feature_extractor(refined)
            
            # Apply temporal consistency
            if features:
                prev_feat = features[-1]
                refined = self._align_with_previous(
                    refined, feat, prev_feat, pose
                )
                
            refined_depths.append(refined)
            features.append(feat)
            
        return refined_depths