#include "UNet.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>

UNet::UNet(int input_channels, int output_channels, int base_filters)
	: Model(), input_channels_(input_channels), output_channels_(output_channels), base_filters_(base_filters)
{
}

void UNet::buildBackboneUNet(int stages, int blocks_per_stage, int bottleneck_depth)
{
	std::cout << "🏗️  Construction UNet: " << stages << " stages, "
			  << blocks_per_stage << " blocks/stage, bottleneck=" << bottleneck_depth << std::endl;

	buildLayers(stages);

	std::cout << "✓ UNet construit avec succès - Total params: "
			  << totalParamCount() << std::endl;
}

void UNet::buildLayers(int stages)
{
	int current_filters = base_filters_;

	// Encodeur: stages niveaux avec downsampling
	for (int stage = 0; stage < stages; ++stage)
	{
		int in_ch = (stage == 0) ? input_channels_ : current_filters / 2;
		int out_ch = current_filters;

		// Conv 3x3: in_ch -> out_ch
		int conv1_params = 3 * 3 * in_ch * out_ch;
		push("encoder_conv1_s" + std::to_string(stage), "Conv2d", conv1_params + out_ch);

		// BatchNorm + Conv 3x3: out_ch -> out_ch
		int conv2_params = 3 * 3 * out_ch * out_ch;
		push("encoder_bn1_s" + std::to_string(stage), "BatchNorm2d", out_ch * 2);
		push("encoder_conv2_s" + std::to_string(stage), "Conv2d", conv2_params + out_ch);
		push("encoder_bn2_s" + std::to_string(stage), "BatchNorm2d", out_ch * 2);

		// MaxPool 2x2 (pas de paramètres)
		push("encoder_pool_s" + std::to_string(stage), "MaxPool2d", 0);

		std::cout << "  Encoder stage " << stage << ": "
				  << in_ch << " -> " << out_ch << " channels" << std::endl;

		current_filters *= 2;
	}

	// Bottleneck: 2 convolutions 3x3
	int bottleneck_params = 3 * 3 * current_filters * current_filters;
	push("bottleneck_conv1", "Conv2d", bottleneck_params + current_filters);
	push("bottleneck_bn1", "BatchNorm2d", current_filters * 2);
	push("bottleneck_conv2", "Conv2d", bottleneck_params + current_filters);
	push("bottleneck_bn2", "BatchNorm2d", current_filters * 2);

	std::cout << "  Bottleneck: " << current_filters << " channels" << std::endl;

	// Décodeur: upsampling path
	for (int stage = 0; stage < stages; ++stage)
	{
		int in_ch = current_filters;
		current_filters /= 2;
		int out_ch = current_filters;

		// ConvTranspose 2x2 pour upsampling: in_ch -> out_ch
		int upconv_params = 2 * 2 * in_ch * out_ch;
		push("decoder_upconv_s" + std::to_string(stage), "ConvTranspose2d", upconv_params + out_ch);

		// Après concatenation avec skip: out_ch*2 -> out_ch
		int concat_ch = out_ch * 2;
		int dec_conv1_params = 3 * 3 * concat_ch * out_ch;
		push("decoder_conv1_s" + std::to_string(stage), "Conv2d", dec_conv1_params + out_ch);
		push("decoder_bn1_s" + std::to_string(stage), "BatchNorm2d", out_ch * 2);

		int dec_conv2_params = 3 * 3 * out_ch * out_ch;
		push("decoder_conv2_s" + std::to_string(stage), "Conv2d", dec_conv2_params + out_ch);
		push("decoder_bn2_s" + std::to_string(stage), "BatchNorm2d", out_ch * 2);

		std::cout << "  Decoder stage " << stage << ": "
				  << in_ch << " -> " << out_ch << " channels" << std::endl;
	}

	// Couche de sortie finale: conv 1x1
	int output_params = 1 * 1 * base_filters_ * output_channels_;
	push("output_conv", "Conv2d", output_params + output_channels_);

	std::cout << "  Output layer: " << base_filters_ << " -> "
			  << output_channels_ << " channels" << std::endl;
}
