#pragma once

#include "../Model.hpp"
#include <vector>
#include <string>

/**
 * @brief UNet Architecture pour la segmentation et génération d'images
 *
 * Architecture U-Net avec encodeur, bottleneck et décodeur avec skip connections
 */
class UNet : public Model
{
public:
	UNet(int input_channels = 3, int output_channels = 3, int base_filters = 64);

	// Override de la construction du backbone
	void buildBackboneUNet(int stages, int blocks_per_stage, int bottleneck_depth) override;

	// Getters
	int getInputChannels() const { return input_channels_; }
	int getOutputChannels() const { return output_channels_; }
	int getBaseFilters() const { return base_filters_; }

private:
	int input_channels_;
	int output_channels_;
	int base_filters_;

	void buildLayers(int stages);
};
