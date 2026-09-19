{
  description = "Ambiente do TCC de geracao de manuscrito (DiffusionPen + BRESSAY)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";

      # Os tres shells diferem so na aceleracao: o conjunto de pacotes Python
      # e identico. Cada membro entra no shell da sua maquina.
      mkPkgs =
        extraConfig:
        import nixpkgs {
          inherit system;
          config = { allowUnfree = true; } // extraConfig;
        };

      pythonPacotes =
        ps: with ps; [
          torch
          torchvision
          numpy
          pillow
          scikit-image
          tqdm
          einops
          transformers
          diffusers
          wandb
          timm
          opencv4 # equivalente ao opencv-python no nixpkgs
          scikit-learn
          matplotlib
          omegaconf
          accelerate
        ];

      mkShell =
        {
          pkgs,
          nome,
          hook ? "",
        }:
        pkgs.mkShell {
          packages = [ (pkgs.python3.withPackages pythonPacotes) ];
          shellHook = ''
            export LD_LIBRARY_PATH="$NIX_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"
            echo "ambiente: ${nome}"
            ${hook}
            if [ -n "$(command -v zsh)" ]; then
              export SHELL="$(command -v zsh)"
              exec zsh
            fi
          '';
        };

      pkgsRocm = mkPkgs { rocmSupport = true; };
      pkgsCuda = mkPkgs { cudaSupport = true; };
      pkgsCpu = mkPkgs { };
    in
    {
      devShells.${system} = {
        # AMD. HSA_OVERRIDE_GFX_VERSION mapeia placas sem kernels proprios para
        # uma arquitetura suportada -- 10.3.0 vale para RDNA2 (RX 6000).
        # Ajuste para a sua placa; numa RDNA3/RDNA4 o valor e outro.
        #
        # ATENCAO: a RX 6600 XT em que este projeto foi desenvolvido NAO computa
        # o modelo de forma confiavel. Rode diagnostico/ antes de treinar em
        # qualquer placa AMD.
        rocm = mkShell {
          pkgs = pkgsRocm;
          nome = "ROCm (AMD)";
          hook = ''
            export HSA_OVERRIDE_GFX_VERSION=''${HSA_OVERRIDE_GFX_VERSION:-10.3.0}
            export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
            echo "  HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
            echo "  rode 'python diagnostico/teste_conv_isolada.py' antes de treinar"
          '';
        };

        # NVIDIA. E o caminho recomendado para treinar.
        # A primeira entrada pode compilar por horas se o cache binario do
        # cuda-maintainers nao estiver configurado -- veja o README.
        cuda = mkShell {
          pkgs = pkgsCuda;
          nome = "CUDA (NVIDIA)";
          hook = ''
            echo "  rode 'python diagnostico/teste_conv_isolada.py' antes de treinar"
          '';
        };

        # Sem GPU. Serve para a parte da metrica que nao gera imagem e para
        # usar a CPU como referencia numerica nos testes de diagnostico.
        cpu = mkShell {
          pkgs = pkgsCpu;
          nome = "CPU";
        };

        # Mantem 'nix develop' funcionando na maquina original.
        default = self.devShells.${system}.rocm;
      };
    };
}
