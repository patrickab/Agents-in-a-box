"""
Programmatically manages all dockerfiles for this application.
Ensures a consistent base layer and isolated agent-specific layers.
"""

import logging
import os

import docker

from code_agents.lib.config import DOCKERFILE_BASE, DOCKERFILE_DEFINITIONS, DOCKERFILES_PATH, DOCKERTAG_BASE, PROJECT_ROOT


class DockerImageBakery:
    def __init__(self) -> None:
        self.client = docker.from_env()
        self.logger = logging.getLogger("ImageFactory")
        logging.basicConfig(level=logging.INFO)

    def build_all(self) -> None:
        """Orchestrates the build process in the correct order."""
        os.makedirs(DOCKERFILES_PATH, exist_ok=True)

        # 1. Build Base Dockerfile
        self._build_image(tag=DOCKERTAG_BASE, dockerfile_content=DOCKERFILE_BASE, target_name="base")

        # 2. Build Dockerfiles from definitions
        for tag, dockerfile_content in DOCKERFILE_DEFINITIONS.items():
            self._build_image(tag=f"{tag}:latest", dockerfile_content=dockerfile_content, target_name=tag)

        self.logger.info("✅ All images baked successfully.")

    def _build_image(self, tag: str, dockerfile_content: str, target_name: str) -> None:
        """
        Writes Dockerfile to disk and builds it.
        """
        # Save Dockerfile for debugging/transparency
        dockerfile_path = os.path.join(DOCKERFILES_PATH, f"Dockerfile.{target_name}")
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)

        self.logger.info(f"Building {tag}...")

        try:
            # Low-level API to stream logs during build
            rel_dockerfile = os.path.relpath(dockerfile_path, PROJECT_ROOT)
            build_logs = self.client.api.build(path=PROJECT_ROOT, dockerfile=rel_dockerfile, tag=tag, rm=True, decode=True)

            for chunk in build_logs:
                if "stream" in chunk:
                    print(chunk["stream"], end="")
                if "error" in chunk:
                    raise docker.errors.BuildError(chunk["error"], build_log=build_logs)
            self.logger.info(f"✅ Built {tag} successfully.")
        except docker.errors.BuildError as e:
            self.logger.error(f"❌ Build failed for {tag}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ System error: {e}")
            raise


if __name__ == "__main__":
    factory = DockerImageBakery()
    factory.build_all()
