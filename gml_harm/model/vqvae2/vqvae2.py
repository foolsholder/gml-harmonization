import torch

from torch import nn

from .vqvae_entities import Encoder, Decoder, Quantize


class HVQEncoder(nn.Module):
    def __init__(self,
                 in_channel=3,
                 channel=128,
                 n_res_block=2,
                 n_res_channel=32,
                 embed_dim=64,
                 n_embed=512,
                 decay=0.99):
        super(HVQEncoder, self).__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed, decay=decay)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )

    def forward(self, x):
        enc_b = self.enc_b(x)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)

        batch_size = x.size(0)
        enc_t = enc_t.view(batch_size, -1)
        enc_b = enc_b.view(batch_size, -1)
        enc_feat = torch.cat([enc_t, enc_b], dim=1)

        return quant, diff_t + diff_b, enc_feat


class HVQVAE(nn.Module):
    """
    Harmonization VQ-VAE
    """
    def __init__(
        self,
        in_channel=3,
        out_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        self.content_enc = HVQVAE(
            in_channel=in_channel,
            channel=channel,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
            embed_dim=embed_dim,
            n_embed=n_embed,
            decay=decay,
        )

        self.reference_enc = HVQVAE(
            in_channel=in_channel,
            channel=channel,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
            embed_dim=embed_dim,
            n_embed=n_embed,
            decay=decay,
        )

        self.dec = Decoder(
            embed_dim + embed_dim + embed_dim + embed_dim,
            out_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, content_input, reference_input):
        content_quant, reference_quant, diff = self.encode(content_input, reference_input)
        dec = self.decode(content_quant, reference_quant)
        return dec, diff

    def encode(self, content_input, reference_input):
        content_quant, content_diff, _ = self.content_enc(content_input)
        reference_quant, reference_diff, _ = self.reference_enc(reference_input)
        return content_quant, reference_quant, content_diff + reference_diff

    def decode(self, content_quant, reference_quant):
        quant = torch.cat([content_quant, reference_quant], 1)
        dec = self.dec(quant)
        return dec