#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
follow_me.py — Suivi de visage avec une caméra PTZ (V4L2).

Compatibilité caméra virtuelle : la disponibilité et la compatibilité de
v4l2loopback dépendent du noyau, de la configuration NixOS et du paquet choisi.
Vérifier les versions installées sur la machine cible ; ce script ne présume pas
qu'un module soit disponible ni qu'il fonctionne pour un noyau donné.


UTILISATION
-----------

Sur NixOS, lancer simplement le script : si des prérequis Python ou des
commandes système manquent, le script écrit le shell.nix intégré dans un fichier
temporaire puis se relance dans nix-shell. Sur Debian, les dépendances sont
vérifiées et les paquets manquants sont indiqués sans installation automatique.
Le marqueur
FOLLOW_ME_IN_NIX_SHELL empêche une relance infinie.

    python3 follow_me.py --no-reset --max-delta 200 --preview

Télécharger le modèle YuNet dans le dossier du script :

    curl -L -o face_detection_yunet_2023mar.onnx \
      https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx

Si plusieurs caméras PTZ sont détectées :

    ls -l /dev/v4l/by-id/
    python3 follow_me.py --device /dev/v4l/by-id/TON-PERIPHERIQUE-video-index0 \
      --no-reset --max-delta 200 --preview

Au lancement, le script réinitialise les moteurs (désactiver avec
--no-start-reset), puis ignore 12 images pour stabiliser l'exposition.
--no-reset désactive séparément le recentrage initial logiciel.
Le zoom automatique est actif par défaut (--no-auto-zoom pour le couper) ;
la cible est réglable avec --zoom-target-ratio.
Le tilt est inversé par défaut. Utiliser --no-invert-tilt pour désactiver
l'inversion ou --invert-pan pour inverser le pan.

CAMÉRA VIRTUELLE (VirtualVideo)
-------------------------------


Une seule application peut ouvrir la vraie caméra à la fois. Avec
VirtualVideo (activée via --virtual-video, l'entrée VirtualVideo du menu
tray ou la case "VirtualVideo (partager le flux)" des réglages), le script
recopie le flux capturé vers une caméra virtuelle v4l2loopback : les
autres logiciels (VLC, Cheese, navigateur, Zoom…) lisent alors la copie
tandis que le script garde la vraie caméra pour le pilotage PTZ.

Prérequis : le module kernel v4l2loopback doit être chargé avec un
périphérique configuré. Exemple NixOS à adapter dans configuration.nix :

    boot.extraModulePackages = [ config.boot.kernelPackages.v4l2loopback ];
    boot.kernelModules = [ "v4l2loopback" ];
    boot.extraModprobeConfig = ''
      options v4l2loopback devices=1 video_nr=10 card_label="CamPartagee" exclusive_caps=1
    '';

Cet attribut de paquet peut ne pas exister pour chaque noyau/canal NixOS :
vérifie sa présence et sa compatibilité avec le noyau effectivement sélectionné
avant d'ajouter la configuration. Ce script ne vérifie pas le catalogue NixOS
et ne peut garantir que le module soit disponible ou chargeable sur ta machine.
Reconstruire et redémarrer (ou temporairement : sudo modprobe
v4l2loopback devices=1 video_nr=10 card_label="CamPartagee"
exclusive_caps=1). Le module doit être chargé avant le démarrage de la
session graphique et donc avant le service utilisateur follow-me. Ajouter
aussi l'utilisateur au groupe video (NixOS :
users.users.<nom>.extraGroups = [ "video" ];), puis fermer/réouvrir la
session pour que le gestionnaire systemd utilisateur hérite du groupe.
Les autres applications ouvrent ensuite /dev/video10 :

    vlc v4l2:///dev/video10

Si l'option est demandée mais qu'aucun périphérique loopback n'est
trouvé, un avertissement détaillé (bloc d'explication avec la
configuration NixOS ci-dessus) est affiché et le suivi continue sans
partage. Le nœud loopback est détecté automatiquement ; --virtual-device
permet de l'imposer.

Diagnostic :

    v4l2-ctl --device /dev/v4l/by-id/TON-PERIPHERIQUE-video-index0 --list-ctrls
    v4l2-ctl --list-devices

Important : la copie ne libère pas le nœud physique. Follow Me garde la
caméra physique ouverte pour capturer et piloter ; si WhatsApp ouvre ce même
nœud, « caméra occupée » est attendu. Dans WhatsApp, sélectionner explicitement
la caméra virtuelle « CamPartagee » (/dev/video10 dans l'exemple), jamais la
caméra physique. Avec exclusive_caps=1, le loopback n'annonce sa capacité de
capture qu'une fois le producteur démarré : attendre le message « VirtualVideo
active », puis fermer/réouvrir le sélecteur caméra de WhatsApp et choisir la
caméra virtuelle. Si elle n'apparaît pas, vérifier dans
`v4l2-ctl --list-devices` que /dev/video10 est bien le loopback et que le
navigateur/WhatsApp a le droit d'accéder aux périphériques vidéo (portail /
permissions de sandbox éventuels). Certains pilotes n'acceptent pas non plus
un descripteur de contrôles PTZ et une capture simultanés ; dans ce cas ce
pilote/cette caméra ne permet pas cette combinaison.

Arrêt : Ctrl+C ou touche q lorsque la prévisualisation est active.

# Installer + démarrer immédiatement (redémarre avec chaque session NixOS) :
python3 follow_me.py --install-service

# Avec des options persistantes (elles sont figées dans l'unité) :
python3 follow_me.py --install-service --no-tray --no-reset --max-delta 200

# Vérifier :
systemctl --user status follow-me
journalctl --user -u follow-me -f

# Désinstaller :
python3 follow_me.py --uninstall-service

"""

import os
os.environ.setdefault("PYSTRAY_BACKEND", "appindicator")

import argparse
import ctypes
import configparser
import errno
import fcntl
import getpass
import glob
import importlib.util
import json
import queue
import math
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import tempfile

APT_PACKAGES = {
    "module Python cv2 (OpenCV)": "python3-opencv",
    "cv2.FaceDetectorYN (OpenCV récent avec YuNet)": "python3-opencv",
    "module Python numpy": "python3-numpy",
    "v4l2-ctl (v4l-utils)": "v4l-utils",
    "ffmpeg": "ffmpeg",
    "curl (téléchargement du modèle YuNet)": "curl",
    "module Python Pillow (PIL)": "python3-pil",
    "module Python tkinter": "python3-tk",
    "module Python pystray": "python3-pystray",
}

# External UI translations live beside this script (safe for NixOS deployments).
LANGUAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_driver.lng")
CURRENT_LANGUAGE = "en"
_TRANSLATIONS = {}

def load_translations():
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        parser.read(LANGUAGE_PATH, encoding="utf-8")
        return {lang: dict(parser.items(lang)) for lang in ("en", "fr") if parser.has_section(lang)}
    except (OSError, configparser.Error):
        return {}

def tr(key, **kwargs):
    """Translate a UI key, falling back to English and then the key itself."""
    text = _TRANSLATIONS.get(CURRENT_LANGUAGE, {}).get(
        key, _TRANSLATIONS.get("en", {}).get(key, key))
    try:
        return text.format(**kwargs)
    except (KeyError, ValueError):
        return text

_TRANSLATIONS = load_translations()

# Dernière image d’aperçu destinée au thread Tkinter.
PREVIEW_QUEUE = queue.Queue(maxsize=1)

try:
    import cv2
except ImportError:
    cv2 = None  # vérifié par check_startup_prerequisites()

try:
    import numpy as np
except ImportError:
    np = None


# Linux videodev2.h
VIDIOC_QUERYCTRL = 0xC0445624
VIDIOC_G_CTRL = 0xC008561B
VIDIOC_S_CTRL = 0xC008561C

PAN_REL = 0x009A0904
TILT_REL = 0x009A0905
PAN_RESET = 0x009A0906
TILT_RESET = 0x009A0907
PAN_ABS = 0x009A0908
TILT_ABS = 0x009A0909
ZOOM_ABS = 0x009A090D

FLAG_DISABLED = 0x0001
FLAG_READ_ONLY = 0x0004
FLAG_INACTIVE = 0x0010
FLAG_WRITE_ONLY = 0x0040

MODEL_NAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_NAME)


def download_model(model_path):
    """Télécharge le modèle YuNet dans un fichier temporaire, puis le valide."""
    part_path = model_path + ".part"
    downloader = shutil.which("curl") or shutil.which("wget")
    if downloader is None:
        # L'installation peut relancer ce script après avoir ajouté curl.
        offer_apt_install(["curl"])
        downloader = shutil.which("curl") or shutil.which("wget")

    try:
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        if os.path.exists(part_path):
            os.remove(part_path)
        if downloader and os.path.basename(downloader) == "curl":
            command = [downloader, "-L", "--fail", "-o", part_path, MODEL_URL]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, timeout=120)
            if result.returncode != 0:
                detail = result.stderr.strip() or f"code de sortie {result.returncode}"
                print(f"Échec du téléchargement du modèle avec curl : {detail}", file=sys.stderr)
                return False
        elif downloader:
            command = [downloader, "-O", part_path, MODEL_URL]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, timeout=120)
            if result.returncode != 0:
                detail = result.stderr.strip() or f"code de sortie {result.returncode}"
                print(f"Échec du téléchargement du modèle avec wget : {detail}", file=sys.stderr)
                return False
        else:
            import urllib.request
            print("curl et wget absents ; téléchargement du modèle via urllib.")
            urllib.request.urlretrieve(MODEL_URL, part_path)

        if not os.path.isfile(part_path) or os.path.getsize(part_path) <= 100_000:
            size = os.path.getsize(part_path) if os.path.isfile(part_path) else 0
            print(f"Échec : modèle téléchargé incomplet ou trop petit ({size} octets).",
                  file=sys.stderr)
            return False
        os.replace(part_path, model_path)
        print(f"Modèle YuNet téléchargé : {model_path}")
        return True
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"Échec du téléchargement du modèle YuNet : {exc}", file=sys.stderr)
        return False
    finally:
        try:
            if os.path.exists(part_path):
                os.remove(part_path)
        except OSError:
            pass

DETECT_W = 320
DETECT_H = 240
MAX_W = 640
MAX_H = 480

SHELL_NIX = r"""{ pkgs ? import <nixpkgs> {} }:
let
  # Nixpkgs renamed AppIndicator between channels/profiles; select what exists.
  appindicator =
    if pkgs ? libayatana-appindicator then pkgs.libayatana-appindicator
    else if pkgs ? libappindicator-gtk3 then pkgs.libappindicator-gtk3
    else null;
  appindicatorPkgs = pkgs.lib.optional (appindicator != null) appindicator;
  giPath = pkgs.lib.makeSearchPath "lib/girepository-1.0" ([
    pkgs.gtk3 pkgs.glib pkgs.pango pkgs.gdk-pixbuf pkgs.atk pkgs.harfbuzz
    pkgs.gobject-introspection
  ] ++ appindicatorPkgs);
in pkgs.mkShell {
  packages = [
    # Keep Python modules on this interpreter's sys.path; standalone system
    # packages would expose binaries/libraries but not importable Python modules.
    (pkgs.python3.withPackages (ps: [
      (ps.opencv4.override { enableGtk3 = true; })
      ps.numpy ps.pystray ps.pillow ps.tkinter ps.pygobject3
    ]))
    pkgs.tk pkgs.v4l-utils pkgs.gtk3 pkgs.gobject-introspection
    pkgs.ffmpeg
  ] ++ appindicatorPkgs;
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
    stdenv.cc.cc.lib zlib libGL glib gtk3 dbus
    libxkbcommon fontconfig freetype libxcb libx11 libxext libxrender libxi
    libsm libice wayland
  ] ++ appindicatorPkgs) + ":/run/opengl-driver/lib";
  shellHook = ''
    # Keep GI lookup tied to this shell/profile while preserving inherited paths.
    export GI_TYPELIB_PATH="${giPath}''${GI_TYPELIB_PATH:+:$GI_TYPELIB_PATH}"
    export QT_QPA_PLATFORM=xcb
    unset QT_STYLE_OVERRIDE
  '';
}
"""

V4L2LOOPBACK_NIXOS_CONFIG = """\
# Exemple NixOS : vérifier que cet attribut existe pour le noyau sélectionné.
{ config, ... }:
{
boot.extraModulePackages = [ config.boot.kernelPackages.v4l2loopback ];
boot.kernelModules = [ "v4l2loopback" ];
boot.extraModprobeConfig = ''
  options v4l2loopback devices=1 video_nr=10 card_label="CamPartagee" exclusive_caps=1
'';
}
"""


def print_virtual_video_warning():
    """Bloc d'explication complet quand le loopback est absent."""
    print(
        "\n"
        "============================================================\n"
        " AVERTISSEMENT — VirtualVideo indisponible\n"
        "============================================================\n"
        "Aucun périphérique v4l2loopback n'a été détecté.\n"
        "Le suivi continue normalement, mais le flux n'est PAS\n"
        "partagé avec les autres applications (VLC, Cheese,\n"
        "navigateur, Zoom…).\n\n"
        "Une seule application peut ouvrir la vraie caméra à la\n"
        "fois : tant que ce script la détient, les autres logiciels\n"
        "ne peuvent pas y accéder. Pour leur donner une copie du\n"
        "signal, le module kernel v4l2loopback doit être chargé.\n\n"
        "--- Exemple à adapter dans /etc/nixos/configuration.nix ---\n"
        f"{V4L2LOOPBACK_NIXOS_CONFIG}"
        "Vérifie que l'attribut de paquet existe pour le noyau sélectionné.\n"
        "------------------------------------------------\n\n"
        "Puis :\n"
        "  sudo nixos-rebuild switch   # et redémarrer\n"
        "ou temporairement :\n"
        "  sudo modprobe v4l2loopback devices=1 video_nr=10 \\\n"
        "    card_label=\"CamPartagee\" exclusive_caps=1\n\n"
        "Les autres applications ouvriront alors /dev/video10 :\n"
        "  vlc v4l2:///dev/video10\n"
        "Un nœud précis peut être imposé avec --virtual-device.\n"
        "============================================================\n",
        file=sys.stderr,
    )


class V4L2Control(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("value", ctypes.c_int32),
    ]


class V4L2QueryCtrl(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("name", ctypes.c_ubyte * 32),
        ("minimum", ctypes.c_int32),
        ("maximum", ctypes.c_int32),
        ("step", ctypes.c_int32),
        ("default_value", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 2),
    ]


# ----------------------------------------------------------------------
# Bootstrap NixOS
# ----------------------------------------------------------------------

def distro_id():
    """Lit l'identifiant de distribution sans supposer de chemins FHS applicatifs."""
    try:
        with open("/etc/os-release", encoding="utf-8") as stream:
            for line in stream:
                key, separator, value = line.partition("=")
                if separator and key.strip() == "ID":
                    return value.strip().strip('"\'').lower()
    except OSError:
        pass
    return ""


def is_nixos():
    """Détecte NixOS par os-release et ses marqueurs usuels."""
    return (distro_id() == "nixos" or os.path.exists("/etc/NIXOS")
            or os.path.exists("/run/current-system"))


def offer_apt_install(missing):
    """Propose d'installer avec apt les prérequis manquants connus."""
    try:
        import distro
        current_distro = distro.id().lower()
    except (ImportError, AttributeError, OSError):
        current_distro = distro_id()
    if current_distro not in ("debian", "ubuntu", "linuxmint", "pop") or shutil.which("apt") is None:
        return False

    packages = set()
    for error in missing:
        for label, package in APT_PACKAGES.items():
            pillow_import_error = (label == "module Python Pillow (PIL)"
                                   and error.startswith("module Python Pillow : "))
            if error == label or error == package or error.startswith(label + " : ") or pillow_import_error:
                packages.add(package)
                break
    installable = sorted(packages)
    if not installable:
        return False

    print("Paquets apt à installer : " + ", ".join(installable))
    try:
        answer = input("Les installer maintenant avec sudo apt ? [o/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if answer not in ("o", "oui", "y", "yes"):
        return False

    command = ["apt", "install", "-y", *installable]
    if os.geteuid() != 0:
        command.insert(0, "sudo")
    try:
        result = subprocess.run(command)
    except OSError as exc:
        print(f"Échec de l'installation apt : {exc}", file=sys.stderr)
        return False
    if result.returncode != 0:
        print(f"Échec de l'installation apt (code {result.returncode}).", file=sys.stderr)
        return False

    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__), *sys.argv[1:]])


# Noms de paquets fournis uniquement à titre d'aide : l'exécutable est toujours
# recherché dans PATH, car NixOS n'a pas de préfixe /usr/bin garanti.
COMMAND_PACKAGES = {
    "v4l2-ctl": ("v4l-utils", "v4l-utils"),
    "ffmpeg": ("ffmpeg", "ffmpeg"),
    "systemctl": ("systemd", "systemd"),
    "nix-shell": ("nix", "nix"),
    "busctl": ("systemd", "systemd"),
    "dbus-send": ("dbus", "dbus"),
}


def require_command(name):
    """Résout une commande via PATH et explique comment la fournir par distro."""
    executable = shutil.which(name)
    if executable:
        return executable
    apt_package, nix_package = COMMAND_PACKAGES.get(name, (name, name))
    if is_nixos():
        guidance = f"NixOS : ajouter {nix_package} à environment.systemPackages ou utiliser nix-shell -p {nix_package}."
    elif distro_id() in ("debian", "ubuntu", "linuxmint", "pop"):
        guidance = f"Debian/Ubuntu : installer le paquet avec sudo apt install {apt_package}."
    else:
        guidance = f"Installez le paquet fournissant {name} (Debian : sudo apt install {apt_package})."
    raise RuntimeError(f"Commande requise introuvable dans PATH : {name}. {guidance}")


def quick_missing(need_tray=False, need_gui=False, need_virtual_video=False):
    """Prérequis Python/système minimaux, sans afficher de chemin local."""
    missing = []
    if cv2 is None:
        missing.append("module Python cv2 (OpenCV)")
    elif not hasattr(cv2, "FaceDetectorYN"):
        missing.append("cv2.FaceDetectorYN (OpenCV récent avec YuNet)")
    if np is None:
        missing.append("module Python numpy")
    if need_tray:
        # Do not import pystray during the host-Python bootstrap probe: loading
        # its GTK/AppIndicator backend can emit GUI warnings before nix-shell
        # has had a chance to provide the intended runtime. The real import is
        # checked after ensure_nix_environment() in check_startup_prerequisites.
        try:
            if importlib.util.find_spec("pystray") is None:
                missing.append("module Python pystray")
        except (ImportError, ValueError) as exc:
            missing.append(f"module Python pystray : {exc}")
        try:
            from PIL import Image, ImageDraw  # noqa: F401
        except ImportError:
            missing.append("module Python Pillow (PIL)")
        try:
            import tkinter  # noqa: F401
        except ImportError:
            missing.append("module Python tkinter")
    if shutil.which("v4l2-ctl") is None:
        missing.append("v4l2-ctl (v4l-utils)")
    if need_virtual_video and shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")
    if need_gui and cv2 is not None:
        info = cv2.getBuildInformation()
        if not any(line.strip().startswith(("GTK+:", "GTK:", "QT:")) and "YES" in line
                   for line in info.splitlines()):
            missing.append("OpenCV compilé avec un backend graphique (GTK/Qt)")
    return missing


def ensure_nix_environment(need_tray=False, need_gui=False, need_virtual_video=False):
    """Sur NixOS, relance dans le shell Nix intégré si un prérequis manque."""
    missing = quick_missing(need_tray, need_gui, need_virtual_video)
    if not missing or os.environ.get("FOLLOW_ME_IN_NIX_SHELL"):
        return
    if not (is_nixos() and shutil.which("nix-shell")):
        return
    print("Prérequis manquants : " + ", ".join(missing) + ".", file=sys.stderr)
    fd, shell_file = tempfile.mkstemp(prefix="follow-me-", suffix=".nix")
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(SHELL_NIX)
    print("NixOS détecté : lancement du shell de dépendances intégré…", file=sys.stderr)
    script = os.path.abspath(__file__)
    # Resolve python3 inside nix-shell so its withPackages environment supplies
    # cv2/numpy/Pillow/pystray/tkinter. An absolute host-Python path here would
    # bypass the Nix interpreter even though nix-shell correctly populated PATH.
    cmd = "python3 " + " ".join(shlex.quote(arg) for arg in [script, *sys.argv[1:]])
    env = {**os.environ, "FOLLOW_ME_IN_NIX_SHELL": "1"}
    nix_shell = shutil.which("nix-shell")
    if not nix_shell:
        return
    os.execvpe(nix_shell, [nix_shell, shell_file, "--run", cmd], env)


def tray_environment_status():
    """Vérifie la session DBus et le watcher StatusNotifier de Plasma."""
    if not os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
        return False, "aucune session DBus détectée"
    if "kde" not in os.environ.get("XDG_CURRENT_DESKTOP", "").lower() and \
            "plasma" not in os.environ.get("XDG_CURRENT_DESKTOP", "").lower():
        return False, "session Plasma/KDE non détectée"
    probe = shutil.which("busctl")
    if probe:
        try:
            result = subprocess.run([probe, "--user", "status", "org.kde.StatusNotifierWatcher"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    timeout=2)
            if result.returncode == 0:
                return True, "session DBus et StatusNotifier disponibles"
        except (OSError, subprocess.SubprocessError):
            pass
    dbus_send = shutil.which("dbus-send")
    if dbus_send:
        try:
            result = subprocess.run([dbus_send, "--session", "--dest=org.freedesktop.DBus",
                                     "/org/freedesktop/DBus", "org.freedesktop.DBus.ListNames"],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    text=True, timeout=2)
            if result.returncode == 0 and "org.kde.StatusNotifierWatcher" in result.stdout:
                return True, "session DBus et StatusNotifier disponibles"
        except (OSError, subprocess.SubprocessError):
            pass
    return False, "service StatusNotifier/DBus indisponible dans cette session"

# ----------------------------------------------------------------------
# Vérification détaillée des prérequis
# ----------------------------------------------------------------------

def check_prerequisites(args):
    """Vérifie les prérequis et affiche un rapport. Lève RuntimeError si bloquant."""
    errors = []
    warnings = []

    def ok(msg):
        print(f"  [OK]   {msg}")

    def warn(msg):
        print(f"  [WARN] {msg}")
        warnings.append(msg)

    def fail(msg):
        print(f"  [ERREUR] {msg}")
        errors.append(msg)

    print("Vérification des prérequis :")

    ok(f"OpenCV {cv2.__version__}")
    if hasattr(cv2, "FaceDetectorYN"):
        ok("cv2.FaceDetectorYN disponible")
    else:
        fail("cv2.FaceDetectorYN absent : OpenCV >= 4.5.4 requis")

    model = os.path.expanduser(args.model)
    if os.path.isfile(model) and os.path.getsize(model) >= 100_000:
        ok(f"Modèle YuNet : {model}")
    elif download_model(model):
        ok(f"Modèle YuNet : {model}")
    else:
        fail(
            f"Modèle YuNet absent ou invalide : {model} ; téléchargement automatique échoué.\n"
            f"         curl -L --fail -o '{model}' {MODEL_URL}"
        )

    try:
        build = cv2.getBuildInformation()
    except cv2.error:
        build = ""

    def build_flag(name):
        for line in build.splitlines():
            if line.strip().startswith(name + ":"):
                return "YES" in line
        return None

    v4l = build_flag("V4L/V4L2")
    if v4l is False:
        fail("OpenCV compilé sans support V4L2")
    elif v4l:
        ok("Support V4L2 dans OpenCV")

    if args.device:
        dev = os.path.expanduser(args.device)
        if not os.path.exists(dev):
            fail(f"Périphérique introuvable : {dev}")
        elif not os.access(dev, os.R_OK | os.W_OK):
            fail(f"Droits insuffisants sur {dev} (ajoute ton utilisateur au groupe 'video')")
        else:
            ok(f"Périphérique accessible : {dev}")
    else:
        paths = glob.glob("/dev/v4l/by-id/*-video-index*")
        if not paths:
            fail("Aucun nœud /dev/v4l/by-id/*-video-index* (caméra branchée ?)")
        else:
            denied = [p for p in paths if not os.access(p, os.R_OK | os.W_OK)]
            if denied:
                fail("Droits insuffisants sur : " + ", ".join(denied) + " (groupe 'video' ?)")
            else:
                ok(f"{len(paths)} nœud(s) vidéo accessible(s)")

    if shutil.which("v4l2-ctl"):
        ok("v4l2-ctl disponible")
    elif args.no_start_reset:
        warn("v4l2-ctl absent ; les resets matériels et commandes zoom sont indisponibles.")
    else:
        if is_nixos():
            dependency_advice = ("NixOS : ajoute v4l-utils à environment.systemPackages "
                                 "ou lance dans nix-shell -p v4l-utils")
        else:
            dependency_advice = "Debian : installe v4l-utils avec sudo apt install v4l-utils"
        fail("v4l2-ctl absent : " + dependency_advice +
             "; requis pour le reset matériel de démarrage (ou utilise --no-start-reset)")

    if args.no_auto_zoom:
        ok("Zoom automatique désactivé par option")
    elif not shutil.which("v4l2-ctl"):
        warn("Zoom auto désactivé : v4l2-ctl absent.")

    # VirtualVideo : signaler tôt l'absence probable de loopback/ffmpeg.
    if getattr(args, "virtual_video", False):
        if shutil.which("ffmpeg") is None:
            if is_nixos():
                ffmpeg_advice = "ajoute ffmpeg à environment.systemPackages ou utilise nix-shell -p ffmpeg"
            else:
                ffmpeg_advice = "installe ffmpeg avec sudo apt install ffmpeg"
            warn("VirtualVideo : ffmpeg absent ; " + ffmpeg_advice +
                 "; la caméra virtuelle ne pourra pas être alimentée.")
        if find_loopback_device(
                os.path.expanduser(args.virtual_device)
                if getattr(args, "virtual_device", None) else None) is None:
            warn("VirtualVideo : aucun périphérique v4l2loopback détecté ; "
                 "vérifie que le module et son nœud /dev/videoN sont présents, "
                 "ou indique-le avec --virtual-device. Voir le bloc d'explication "
                 "affiché au démarrage du suivi.")

    if args.preview:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            fail("--preview demandé mais aucun environnement graphique (DISPLAY / WAYLAND_DISPLAY vides)")
        else:
            gui = None
            for line in build.splitlines():
                if line.strip().startswith(("GTK+:", "GTK:", "QT:", "Win32 UI:", "Cocoa:")):
                    if "YES" in line:
                        gui = line.strip()
                        break
            if gui is None:
                fail(
                    "OpenCV sans backend graphique (GTK/Qt) : "
                    "utilise opencv4.override { enableGtk3 = true; } "
                    "ou lance sans --preview"
                )
            else:
                ok(f"Backend graphique : {gui}")

    if warnings:
        print(f"  {len(warnings)} avertissement(s).")
    if errors:
        raise RuntimeError(f"{len(errors)} prérequis manquant(s) ; corrige-les puis relance.")
    print("Tous les prérequis bloquants sont satisfaits.\n")


# ----------------------------------------------------------------------
# Utilitaires et contrôle PTZ
# ----------------------------------------------------------------------

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def snap(value, control):
    """Arrondit sur la grille minimum + n * step."""
    minimum, maximum, step = control
    value = clamp(value, minimum, maximum)
    value = minimum + round((value - minimum) / step) * step
    return int(clamp(value, minimum, maximum))


class PTZCamera:
    def __init__(self, path):
        self.path = path
        self.fd = None
        self.last_motor_ts = 0.0
        try:
            self.fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
            self._detect_caps()
        except Exception:
            self.close()
            raise

    def query(self, cid, require_read=False):
        q = V4L2QueryCtrl()
        q.id = cid
        try:
            fcntl.ioctl(self.fd, VIDIOC_QUERYCTRL, q)
        except OSError as exc:
            if exc.errno in (errno.EINVAL, errno.ENOTTY):
                return None
            raise
        forbidden = FLAG_DISABLED | FLAG_READ_ONLY | FLAG_INACTIVE
        if q.flags & forbidden:
            return None
        if require_read and q.flags & FLAG_WRITE_ONLY:
            return None
        return q.minimum, q.maximum, max(1, q.step)

    def get(self, cid):
        ctrl = V4L2Control(cid, 0)
        fcntl.ioctl(self.fd, VIDIOC_G_CTRL, ctrl)
        return ctrl.value

    def set(self, cid, value):
        ctrl = V4L2Control(cid, int(value))
        fcntl.ioctl(self.fd, VIDIOC_S_CTRL, ctrl)

    def _detect_caps(self):
        pan = self.query(PAN_ABS, require_read=True)
        tilt = self.query(TILT_ABS, require_read=True)
        if pan is not None and tilt is not None:
            try:
                pan_value = self.get(PAN_ABS)
                tilt_value = self.get(TILT_ABS)
            except OSError:
                pass
            else:
                self.absolute = True
                self.mode = "absolu"
                self.pan_range = pan
                self.tilt_range = tilt
                self.pan_target = float(pan_value)
                self.tilt_target = float(tilt_value)
                self.pan_sent = pan_value
                self.tilt_sent = tilt_value
                return
        pan = self.query(PAN_REL)
        tilt = self.query(TILT_REL)
        if pan is None or tilt is None:
            raise RuntimeError(
                f"{self.path} : contrôles pan/tilt absolus lisibles ou relatifs indisponibles."
            )
        self.absolute = False
        self.mode = "relatif"
        self.pan_range = pan
        self.tilt_range = tilt
        self.pan_pending = 0.0
        self.tilt_pending = 0.0

    def refresh_absolute(self):
        if self.absolute:
            self.pan_sent = self.get(PAN_ABS)
            self.tilt_sent = self.get(TILT_ABS)
            self.pan_target = float(self.pan_sent)
            self.tilt_target = float(self.tilt_sent)

    def center(self):
        if self.absolute:
            pmin, pmax, _ = self.pan_range
            tmin, tmax, _ = self.tilt_range
            pan = snap((pmin + pmax) / 2, self.pan_range)
            tilt = snap((tmin + tmax) / 2, self.tilt_range)
            self.set(PAN_ABS, pan)
            self.set(TILT_ABS, tilt)
            self.pan_sent = pan
            self.tilt_sent = tilt
            self.pan_target = float(pan)
            self.tilt_target = float(tilt)
        else:
            if self.query(PAN_RESET) is None or self.query(TILT_RESET) is None:
                return False
            self.set(PAN_RESET, 1)
            self.set(TILT_RESET, 1)
            self.pan_pending = 0.0
            self.tilt_pending = 0.0
        self.last_motor_ts = time.monotonic()
        return True

    def move(self, pan_delta, tilt_delta, min_interval, max_delta,
             invert_pan=False, invert_tilt=True):
        pan_delta = -pan_delta if invert_pan else pan_delta
        tilt_delta = -tilt_delta if invert_tilt else tilt_delta
        now = time.monotonic()
        if now - self.last_motor_ts < min_interval:
            return False
        pan_delta = clamp(pan_delta, -max_delta, max_delta)
        tilt_delta = clamp(tilt_delta, -max_delta, max_delta)
        if self.absolute:
            pmin, pmax, _ = self.pan_range
            tmin, tmax, _ = self.tilt_range
            self.pan_target = clamp(self.pan_target + pan_delta, pmin, pmax)
            self.tilt_target = clamp(self.tilt_target + tilt_delta, tmin, tmax)
            pan = snap(self.pan_target, self.pan_range)
            tilt = snap(self.tilt_target, self.tilt_range)
            changed = False
            if pan != self.pan_sent:
                self.set(PAN_ABS, pan)
                self.pan_sent = pan
                changed = True
            if tilt != self.tilt_sent:
                self.set(TILT_ABS, tilt)
                self.tilt_sent = tilt
                changed = True
        else:
            self.pan_pending = clamp(self.pan_pending + pan_delta, -max_delta, max_delta)
            self.tilt_pending = clamp(self.tilt_pending + tilt_delta, -max_delta, max_delta)
            pan = snap(self.pan_pending, self.pan_range)
            tilt = snap(self.tilt_pending, self.tilt_range)
            changed = False
            if pan:
                self.set(PAN_REL, pan)
                self.pan_pending -= pan
                changed = True
            if tilt:
                self.set(TILT_REL, tilt)
                self.tilt_pending -= tilt
                changed = True
        if changed:
            self.last_motor_ts = now
        return changed

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def select_camera(device):
    if device:
        return PTZCamera(device)
    paths = sorted(glob.glob("/dev/v4l/by-id/*-video-index*"))
    preferred = "/dev/v4l/by-id/usb-046d_08b5-video-index0"
    if preferred in paths:
        paths.remove(preferred)
        paths.insert(0, preferred)
    candidates = []
    errors = []
    seen = set()
    for path in paths:
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        try:
            candidates.append(PTZCamera(path))
        except (OSError, RuntimeError) as exc:
            errors.append(f"  {path} : {exc}")
    if len(candidates) == 1:
        return candidates[0]
    for camera in candidates:
        camera.close()
    if len(candidates) > 1:
        choices = "\n".join(f"  {camera.path}" for camera in candidates)
        raise RuntimeError("Plusieurs nœuds PTZ détectés ; choisis-en un avec --device :\n" + choices)
    detail = "\n".join(errors) if errors else "  Aucun nœud by-id trouvé."
    raise RuntimeError(
        "Aucune caméra PTZ utilisable détectée.\n"
        f"{detail}\n"
        "Vérifie les droits et les contrôles avec v4l2-ctl, ou indique le nœud avec --device."
    )


def query_zoom_control(device):
    try:
        result = subprocess.run(
            [require_command("v4l2-ctl"), "-d", device, "--list-ctrls"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        detail = getattr(exc, "stderr", None)
        return None, (detail.strip() if detail else str(exc))
    controls = {}
    for line in result.stdout.splitlines():
        match = re.match(
            r"^\s*(zoom_absolute|zoom_continuous|zoom_relative)"
            r"(?:\s+0x[0-9a-fA-F]+)?\s+\(([^)]*)\)\s*:", line
        )
        if not match:
            continue
        name, kind = match.groups()
        values = {key: int(value) for key, value in
                  re.findall(r"(min|max|step|default|value)=(-?\d+)", line)}
        flags_match = re.search(r"\bflags=([^ ]+)", line)
        flags = flags_match.group(1).strip("[]") if flags_match else ""
        controls[name] = (kind, values, flags, line.strip())
    for name in ("zoom_absolute", "zoom_continuous", "zoom_relative"):
        if name not in controls:
            continue
        kind, values, flags, line = controls[name]
        if "disabled" in flags or "inactive" in flags:
            continue
        if not all(key in values for key in ("min", "max", "step")):
            continue
        return {
            "name": name, "kind": kind,
            "minimum": values["min"], "maximum": values["max"],
            "step": max(1, abs(values["step"])),
            "default": values.get("default", values["min"]),
            "relative": name != "zoom_absolute",
        }, None
    if controls:
        details = "; ".join(f"{name}: {info[2] or 'bornes indisponibles'}"
                             for name, info in controls.items())
        return None, "contrôle zoom présent mais inutilisable (inactif/désactivé ou sans bornes): " + details
    return None, "aucun zoom_absolute, zoom_continuous ou zoom_relative dans --list-ctrls"


class ZoomController:
    """Zoom V4L2 borné, temporisé et piloté par le ratio de taille du visage."""

    def __init__(self, device, enabled=True, target_ratio=0.25,
                 max_step=None, max_zoom=3.0, min_interval=0.12, smoothing=0.55):
        self.device = device
        self.info, error = query_zoom_control(device)
        self.enabled = bool(enabled and self.info is not None)
        self.target_ratio = target_ratio
        self.min_interval = min_interval
        self.smoothing = smoothing
        self.filtered_ratio = None
        self.last_command_at = 0.0
        self.current = None
        self.digital = bool(enabled and self.info is None)
        self.max_zoom = float(max_zoom)
        self.control = self.info["name"] if self.info else None
        if self.info is None:
            self.current = 1.0
            self.digital = True
            self.enabled = bool(enabled)
            print("Zoom auto : zoom numérique (aucun contrôle V4L2 zoom).")
        else:
            bounds = f"{self.info['minimum']}..{self.info['maximum']}"
            print(f"Zoom auto : zoom matériel trouvé — contrôle {self.control}, plage {bounds}, pas {self.info['step']}.")
        if not enabled:
            print("Zoom auto : désactivé par --no-auto-zoom.")
        if self.info is not None:
            self.minimum = self.info["minimum"]
            self.maximum = self.info["maximum"]
            self.control_step = self.info["step"]
            self.max_step = max_step if max_step is not None else 8 * self.control_step
            self.max_step = max(self.control_step,
                                int(round(self.max_step / self.control_step)) * self.control_step)
            if self.info["relative"]:
                if not (self.minimum < 0 < self.maximum):
                    print(f"Zoom auto désactivé : {self.control} ne fournit pas de plage bidirectionnelle exploitable.", file=sys.stderr)
                    self.enabled = False
                    self.current = None
                elif enabled:
                    self.current = 0
            elif enabled:
                self._read_current()

    def _read_current(self):
        try:
            readback = subprocess.run(
                [require_command("v4l2-ctl"), "-d", self.device, f"--get-ctrl={self.control}"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=5,
            )
            match = re.search(rf"{re.escape(self.control)}\s*:\s*(-?\d+)", readback.stdout)
            if not match:
                raise RuntimeError(f"valeur {self.control} absente de la réponse")
            self.current = int(match.group(1))
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            detail = getattr(exc, "stderr", None)
            print(f"[AVERTISSEMENT] Lecture de {self.control} impossible : "
                  f"{detail.strip() if detail else exc}; zoom auto désactivé.", file=sys.stderr)
            self.enabled = False

    def _set_value(self, value, ratio=None):
        relative = self.info["relative"]
        if relative:
            value = max(self.minimum, min(self.maximum, int(value)))
            if value == 0:
                return False
        else:
            highest_valid = self.minimum + ((self.maximum - self.minimum) // self.control_step) * self.control_step
            value = clamp(snap(value, (self.minimum, self.maximum, self.control_step)),
                          self.minimum, highest_valid)
            if value == self.current:
                return False
        command = [require_command("v4l2-ctl"), "-d", self.device, f"--set-ctrl={self.control}={value}"]
        try:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            detail = getattr(exc, "stderr", None) or getattr(exc, "stdout", None)
            print(f"[AVERTISSEMENT] Commande {' '.join(command)} échouée : "
                  f"{detail.strip() if detail else exc}", file=sys.stderr)
            self.enabled = False
            return False
        if not relative:
            self.current = value
        self.last_command_at = time.monotonic()
        ratio_text = f" ratio={ratio:.3f}" if ratio is not None else ""
        print(f"Zoom envoyé : contrôle={self.control}{ratio_text} "
              f"cible={self.target_ratio:.3f} valeur={value}" +
              (" (relatif)" if relative else ""))
        return True

    def reset(self):
        if self.info is None or self.current is None or self.info["relative"]:
            return False
        return self._set_value(self.info.get("default", self.minimum))

    def update(self, face_ratio):
        if not self.enabled or self.current is None:
            return False
        if self.filtered_ratio is None:
            self.filtered_ratio = face_ratio
        else:
            a = self.smoothing
            self.filtered_ratio = a * face_ratio + (1 - a) * self.filtered_ratio
        low = self.target_ratio * 0.9
        high = self.target_ratio * 1.1
        if low <= self.filtered_ratio <= high:
            return False
        now = time.monotonic()
        if now - self.last_command_at < self.min_interval:
            return False
        direction = 1 if self.filtered_ratio < low else -1
        ratio = self.filtered_ratio
        if self.digital:
            target = clamp(self.current * self.target_ratio / max(ratio, 1e-6), 1.0, self.max_zoom)
            delta = self.smoothing * (target - self.current)
            delta = clamp(delta, -0.08, 0.08)
            if abs(delta) < 0.001:
                return False
            self.current = clamp(self.current + delta, 1.0, self.max_zoom)
            self.last_command_at = now
            print(f"Zoom numérique : ratio={ratio:.3f} facteur={self.current:.3f}")
            return True
        if self.info["relative"]:
            value = self.maximum if direction > 0 else self.minimum
        else:
            value = clamp(self.current + direction * self.max_step, self.minimum, self.maximum)
        return self._set_value(value, ratio=ratio)

    def update_lost(self, lost_seconds):
        if not self.enabled or not self.digital or self.current <= 1.0:
            return False
        now = time.monotonic()
        if lost_seconds <= 5.0 or now - self.last_command_at < self.min_interval:
            return False
        self.current = max(1.0, self.current - 0.02)
        self.last_command_at = now
        return True


def crop_center_zoom(frame, factor):
    factor = max(1.0, float(factor))
    if factor <= 1.0:
        return frame.copy()
    height, width = frame.shape[:2]
    crop_w = max(1, int(round(width / factor)))
    crop_h = max(1, int(round(height / factor)))
    x0 = (width - crop_w) // 2
    y0 = (height - crop_h) // 2
    cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def reset_camera(device, wait=3.0):
    command = [require_command("v4l2-ctl"), "-d", device, "--set-ctrl=pan_reset=1,tilt_reset=1"]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", None)
        detail = detail.strip() if detail else str(exc)
        print(f"[AVERTISSEMENT] Reset initial non exécuté par v4l2-ctl : {detail}", file=sys.stderr)
        return False
    print(f"Reset pan/tilt envoyé ; stabilisation {wait:g} s…")
    time.sleep(max(0.0, wait))
    return True


# ----------------------------------------------------------------------
# Caméra virtuelle (v4l2loopback) — VirtualVideo
# ----------------------------------------------------------------------

def is_loopback_device(video_node):
    """Confirme par sysfs qu'un chemin /dev/videoN est bien un v4l2loopback."""
    base = "/sys/class/video4linux/" + os.path.basename(os.path.realpath(video_node))
    virtual_root = os.path.realpath("/sys/devices/virtual/video4linux")
    try:
        with open(os.path.join(base, "name"), encoding="utf-8") as stream:
            name = stream.read().strip().lower()
        sysfs_path = os.path.realpath(base)
        is_virtual = os.path.commonpath((virtual_root, sysfs_path)) == virtual_root
    except (OSError, ValueError):
        return False
    return is_virtual or "v4l2 loopback" in name


def find_loopback_device(preferred=None):
    """Trouve un nœud loopback, et refuse une cible explicite physique."""
    if preferred:
        return preferred if os.path.exists(preferred) and is_loopback_device(preferred) else None

    # Le champ sysfs "name" est le libellé de la carte (souvent personnalisé
    # avec card_label), pas le nom du pilote. Il n'inclut donc pas forcément
    # "v4l2 loopback". Les devices de v4l2loopback sont aussi publiés sous
    # /sys/devices/virtual/video4linux ; on reconnaît ce chemin indépendamment
    # du nom de carte. Le test du libellé conserve la compatibilité historique.
    virtual_root = os.path.realpath("/sys/devices/virtual/video4linux")
    for base in sorted(glob.glob("/sys/class/video4linux/video*")):
        video_node = "/dev/" + os.path.basename(base)
        if not os.path.exists(video_node):
            continue
        if is_loopback_device(video_node):
            return video_node
    return None


class VirtualFeeder:
    """Publish captured BGR frames to the v4l2loopback node via ffmpeg stdin."""

    def __init__(self, virtual, width, height, fps):
        # OpenCV's requested dimensions are advisory; use the first actual frame.
        cmd = [
            require_command("ffmpeg"), "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{width}x{height}", "-framerate", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "rawvideo",
            "-pix_fmt", "yuv420p", "-s", f"{width}x{height}",
            "-r", str(fps), "-f", "v4l2", virtual,
        ]
        self.virtual = virtual
        self.proc = None
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            raise RuntimeError(f"Lancement de ffmpeg impossible : {exc}") from exc
        # Fail fast on unsupported formats, permissions, or unavailable output.
        time.sleep(0.15)
        if self.proc.poll() is not None:
            returncode = self.proc.returncode
            self.stop()
            raise RuntimeError(
                f"ffmpeg a quitté au démarrage (code {returncode}); "
                "vérifie les droits /dev/videoN et le format supporté par v4l2loopback."
            )

    def feed(self, frame):
        if self.proc is None or self.proc.stdin is None:
            return
        try:
            self.proc.stdin.write(frame.tobytes())
            if self.proc.poll() is not None:
                raise BrokenPipeError(f"ffmpeg terminé (code {self.proc.returncode})")
        except (BrokenPipeError, OSError) as exc:
            print(f"[AVERTISSEMENT] Flux virtuel interrompu : {exc}", file=sys.stderr)
            self.stop()

    def stop(self):
        if self.proc is None:
            return
        proc = self.proc
        self.proc = None
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
            proc.wait()


# ----------------------------------------------------------------------
# Capture vidéo
# ----------------------------------------------------------------------

class FreshCamera:
    """Le thread possède VideoCapture et le libère lui-même."""

    def __init__(self, path, width, height, fps):
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.error = None
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.path, cv2.CAP_V4L2)
            if not cap.isOpened():
                raise RuntimeError(f"Impossible d'ouvrir {self.path} en capture.")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            while not self.stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.stop.wait(0.02)
                    continue
                with self.lock:
                    self.frame = frame
                    self.seq += 1
        except Exception as exc:
            with self.lock:
                self.error = exc
        finally:
            if cap is not None:
                cap.release()

    def latest(self, last_seq):
        with self.lock:
            if self.error is not None:
                raise RuntimeError(f"Erreur de capture : {self.error}")
            if self.frame is None or self.seq == last_seq:
                return None, last_seq
            return self.frame.copy(), self.seq

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            print("Attention : le pilote bloque encore une lecture caméra ; le thread de capture n'a pas terminé.", file=sys.stderr)

# Suivi
# ----------------------------------------------------------------------

def axis_step(error, deadzone, gain):
    if abs(error) <= deadzone:
        return 0.0
    effective = error - math.copysign(deadzone, error)
    return gain * effective


def show_preview(frame, face, smooth):
    display = frame.copy()
    height, width = frame.shape[:2]
    sx = width / DETECT_W
    sy = height / DETECT_H

    if face is not None:
        x, y, w, h = face[:4]
        cv2.rectangle(
            display,
            (int(x * sx), int(y * sy)),
            (int((x + w) * sx), int((y + h) * sy)),
            (0, 255, 0),
            2,
        )

    if smooth is not None:
        x = int(width / 2 + smooth[0] * sx)
        y = int(height / 2 + smooth[1] * sy)
        cv2.drawMarker(display, (x, y), (0, 0, 255),
                       cv2.MARKER_CROSS, 20, 2)

    try:
        PREVIEW_QUEUE.put_nowait(display.copy())
    except queue.Full:
        try:
            PREVIEW_QUEUE.get_nowait()
        except queue.Empty:
            pass
        try:
            PREVIEW_QUEUE.put_nowait(display.copy())
        except queue.Full:
            pass
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Suivi de visage PTZ V4L2")
    p.add_argument("-d", "--device", help="nœud V4L2 à utiliser")
    p.add_argument("--model", default=DEFAULT_MODEL, help="chemin du modèle ONNX")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--detect-every", type=int, default=2)
    p.add_argument("--score", type=float, default=0.5)
    p.add_argument("--smooth", type=float, default=0.6)
    p.add_argument("--deadzone", type=float, nargs=2, default=(16, 12),
                   metavar=("X", "Y"))
    p.add_argument("--kp-pan", type=float, default=6.0)
    p.add_argument("--kp-tilt", type=float, default=4.5)
    p.add_argument("--invert-pan", action="store_true", default=False,
                   help="inverser le sens du pan (désactivé par défaut)")
    p.add_argument("--invert-tilt", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="inverser le tilt par défaut ; --no-invert-tilt pour désactiver")
    p.add_argument("--no-start-reset", action="store_true",
                   help="ne pas envoyer pan_reset/tilt_reset au lancement")
    p.add_argument("--motor-interval", type=float, default=0.06)
    p.add_argument("--max-delta", type=float, default=1000.0,
                   help="correction maximale par axe et commande")
    p.add_argument("--lost-recenter", type=float, default=3.0,
                   help="secondes avant recentrage ; 0 désactive")
    p.add_argument("--frame-timeout", type=float, default=5.0)
    p.add_argument("--no-auto-zoom", action="store_true",
                   help="désactiver le zoom automatique (actif par défaut)")
    p.add_argument("--zoom-target-ratio", type=float, default=0.25,
                   help="largeur de visage cible / largeur image (défaut : 0.25)")
    p.add_argument("--zoom-max", type=float, default=3.0,
                   help="facteur maximal du zoom numérique (défaut : 3.0)")
    p.add_argument("--zoom-step", type=float, default=None,
                   help="pas maximal par commande ; défaut : 3 pas du contrôle caméra")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--virtual-video", action="store_true",
                   help="partager le flux vers un périphérique v4l2loopback")
    p.add_argument("--virtual-device",
                   help="nœud loopback à utiliser (détection automatique sinon)")
    p.add_argument("--no-tray", action="store_true",
                   help="mode terminal sans icône ni panneau de réglages")
    p.add_argument("--no-reset", action="store_true",
                   help="ne pas recentrer après la première image ; le reset de démarrage reste actif")
    p.add_argument("--install-service", action="store_true",
                   help="installer le script comme service utilisateur systemd "
                        "(démarrage automatique avec la session NixOS) puis le démarrer")
    p.add_argument("--uninstall-service", action="store_true",
                   help="arrêter et supprimer le service systemd utilisateur")
    args = p.parse_args()

    floats = [
        args.score, args.smooth, *args.deadzone,
        args.kp_pan, args.kp_tilt, args.motor_interval,
        args.max_delta, args.lost_recenter, args.frame_timeout,
        args.zoom_target_ratio, args.zoom_max,
    ]
    if args.zoom_step is not None:
        floats.append(args.zoom_step)
    if not all(math.isfinite(value) for value in floats):
        p.error("les paramètres numériques doivent être finis")
    if not (1 <= args.width <= MAX_W and 1 <= args.height <= MAX_H):
        p.error("la résolution demandée doit être comprise entre 1x1 et 640x480")
    if args.fps < 1 or args.detect_every < 1:
        p.error("--fps et --detect-every doivent être >= 1")
    if not (0 < args.score <= 1 and 0 < args.smooth <= 1):
        p.error("--score et --smooth doivent être compris entre 0 exclu et 1")
    if min(args.deadzone) < 0 or args.kp_pan < 0 or args.kp_tilt < 0:
        p.error("zones mortes et gains doivent être positifs ou nuls")
    if args.motor_interval < 0 or args.lost_recenter < 0:
        p.error("les intervalles doivent être positifs ou nuls")
    if args.max_delta <= 0 or args.frame_timeout <= 0:
        p.error("--max-delta et --frame-timeout doivent être > 0")
    if not 0 < args.zoom_target_ratio < 1:
        p.error("--zoom-target-ratio doit être compris entre 0 et 1")
    if args.zoom_max < 1:
        p.error("--zoom-max doit être >= 1")
    if args.zoom_step is not None and args.zoom_step <= 0:
        p.error("--zoom-step doit être > 0")

    return args


def run_tracking(args, live_settings=None, stop=None):
    """Exécute sans modification la boucle PTZ historique, dans son thread."""
    global cv2
    if cv2 is None:
        try:
            import cv2 as _cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV est absent ; vérifiez les dépendances.") from exc
        cv2 = _cv2
    if live_settings is not None:
        with live_settings["lock"]:
            initial = dict(live_settings["values"])
        args.zoom_max = initial["zoom_max"]
        args.zoom_target_ratio = initial["face_target"]
        args.deadzone = (initial["deadzone_x"], initial["deadzone_y"])
        args.kp_pan = 6.0 * initial["speed"]
        args.kp_tilt = 4.5 * initial["speed"]
        args.invert_tilt = initial["invert_y"]
        args.invert_pan = initial.get("invert_pan", args.invert_pan)
        args.preview = initial["preview"]
        args.no_auto_zoom = not initial["zoom_enabled"]
        args.virtual_video = bool(initial.get("virtual_video", False))
    check_prerequisites(args)
    model = os.path.expanduser(args.model)

    if not os.path.isfile(model):
        raise RuntimeError(f"Modèle YuNet introuvable : {model}")

    detector = cv2.FaceDetectorYN.create(
        model, "", (DETECT_W, DETECT_H), args.score, 0.6, 20
    )

    stop = stop if stop is not None else threading.Event()

    ptz = None
    cam = None
    feeder = None

    try:
        ptz = select_camera(args.device)
        print(
            f"Caméra : {ptz.path} ; mode {ptz.mode} ; "
            f"pan {ptz.pan_range} ; tilt {ptz.tilt_range}"
        )
        zoom = ZoomController(
            ptz.path, enabled=not args.no_auto_zoom,
            target_ratio=args.zoom_target_ratio, max_step=args.zoom_step,
            max_zoom=args.zoom_max,
        )
        if zoom.info is not None and zoom.enabled:
            print(f"Cible zoom : ratio visage/image {args.zoom_target_ratio:.3f}; "
                  f"délai minimal {zoom.min_interval:.2f}s.")

        # Reset moteur distinct du recentrage logiciel --no-reset.
        if not args.no_start_reset:
            reset_camera(ptz.path)
        else:
            print("Reset matériel de démarrage désactivé (--no-start-reset).")
        if zoom.info is not None:
            if zoom.reset():
                print(f"Reset zoom démarrage : {zoom.current}.")
            else:
                print("Zoom déjà à sa valeur de reset.")

        cam = FreshCamera(ptz.path, args.width, args.height, args.fps)
        virtual = None
        feeder_failed = False
        if getattr(args, "virtual_video", False):
            requested_virtual = (os.path.expanduser(args.virtual_device)
                                 if args.virtual_device else None)
            virtual = find_loopback_device(requested_virtual)
            if virtual is None:
                if requested_virtual and os.path.exists(requested_virtual):
                    print(f"[AVERTISSEMENT] {requested_virtual} n'est pas identifié comme v4l2loopback ; "
                          "refus de l'ouvrir en sortie (la caméra source est ainsi protégée).",
                          file=sys.stderr)
                print_virtual_video_warning()
                virtual = None
            elif os.path.realpath(virtual) == os.path.realpath(ptz.path):
                print("[AVERTISSEMENT] La sortie virtuelle désigne la caméra source ; partage désactivé.",
                      file=sys.stderr)
                virtual = None
                feeder_failed = True
            elif shutil.which("ffmpeg") is None:
                print("[AVERTISSEMENT] VirtualVideo demandée mais ffmpeg est introuvable ; partage désactivé.",
                      file=sys.stderr)
                virtual = None
                feeder_failed = True

        last_seq = -1
        last_frame_at = time.monotonic()
        first_frame = True
        frame_index = 0
        warmup_frames_remaining = 12
        startup_search = True
        startup_search_started = time.monotonic()
        search_step_index = 0
        next_search_at = startup_search_started
        last_face = None
        lost_since = None
        recentered_for_loss = False
        smooth_x = 0.0
        smooth_y = 0.0
        has_smooth = False

        print("Suivi actif : Ctrl+C ou q pour quitter.")

        while not stop.is_set():
            if live_settings is not None:
                with live_settings["lock"]:
                    current = dict(live_settings["values"])
                args.zoom_max = current["zoom_max"]
                args.zoom_target_ratio = current["face_target"]
                args.deadzone = (current["deadzone_x"], current["deadzone_y"])
                args.kp_pan = 6.0 * current["speed"]
                args.kp_tilt = 4.5 * current["speed"]
                args.invert_tilt = current["invert_y"]
                args.invert_pan = current.get("invert_pan", args.invert_pan)
                args.preview = current["preview"]
                if "zoom" in locals():
                    zoom.max_zoom = current["zoom_max"]
                    zoom.target_ratio = current["face_target"]
                    was_enabled = zoom.enabled
                    zoom.enabled = bool(current["zoom_enabled"] and
                                        (zoom.info is not None or zoom.digital))
                    if zoom.enabled and not was_enabled and zoom.current is None:
                        if zoom.info is not None and zoom.info["relative"]:
                            zoom.current = 0
                        elif zoom.info is not None:
                            zoom._read_current()
                    if zoom.digital and zoom.current is not None:
                        zoom.current = clamp(zoom.current, 1.0, zoom.max_zoom)
            # Capture the just-refreshed inversion value for this iteration;
            # both PTZ movement paths use this, never the startup setting.
            current_invert_tilt = args.invert_tilt
            frame, seq = cam.latest(last_seq)

            if frame is None:
                if time.monotonic() - last_frame_at > args.frame_timeout:
                    raise RuntimeError(
                        tr("no_new_frame")
                    )
                stop.wait(0.003)
                continue

            last_seq = seq
            last_frame_at = time.monotonic()
            frame_index += 1
            height, width = frame.shape[:2]
            if virtual is not None and feeder is None and not feeder_failed:
                try:
                    feeder = VirtualFeeder(virtual, width, height, args.fps)
                    print(f"VirtualVideo active : copie {width}x{height} vers {virtual}. "
                          "Dans WhatsApp, sélectionner ce périphérique virtuel.")
                except (OSError, RuntimeError) as exc:
                    feeder_failed = True
                    print(f"[AVERTISSEMENT] Démarrage VirtualVideo impossible : {exc}",
                          file=sys.stderr)
            if feeder is not None:
                feeder.feed(frame)

            if width > MAX_W or height > MAX_H:
                raise RuntimeError(
                    f"Résolution réelle {width}x{height} supérieure à "
                    "640x480 : arrêt avant commande PTZ."
                )

            if first_frame:
                first_frame = False
                print(f"Capture réelle : {width}x{height}")

                if not args.no_reset:
                    if ptz.center():
                        print("Recentrage initial…")
                        if stop.wait(1.5):
                            break
                        ptz.refresh_absolute()
                    else:
                        print("Recentrage initial indisponible sur ce pilote.")

            # Les images capturées juste après l'ouverture ont souvent une
            # exposition/ balance des blancs transitoire. Les ignorer évite
            # une commande basée sur une détection initiale instable.
            if warmup_frames_remaining > 0:
                warmup_frames_remaining -= 1
                preview_frame = (crop_center_zoom(frame, zoom.current)
                                 if zoom.digital and zoom.enabled else frame)
                if args.preview and not show_preview(preview_frame, None, None):
                    break
                if warmup_frames_remaining == 0:
                    print("Warm-up terminé (12 images ignorées).")
                continue

            # Digital zoom is applied to the captured frame before both face
            # detection and preview; remember this factor for PTZ calibration.
            detection_zoom = zoom.current if zoom.digital and zoom.enabled else 1.0
            zoomed_frame = (crop_center_zoom(frame, detection_zoom)
                            if zoom.digital and zoom.enabled else frame)
            detected_now = (
                frame_index % args.detect_every == 0 or last_face is None
            )

            if detected_now:
                small = cv2.resize(zoomed_frame, (DETECT_W, DETECT_H))
                _, faces = detector.detect(small)

                if faces is None or len(faces) == 0:
                    last_face = None
                    has_smooth = False
                else:
                    # En présence de plusieurs visages, suivre le plus grand.
                    last_face = max(faces, key=lambda face: face[2] * face[3])

            if last_face is None:
                if lost_since is None:
                    lost_since = time.monotonic()

                # Au démarrage, plutôt que de rester immobile ou de recentrer
                # en boucle, parcourir progressivement l'axe relatif. Les
                # commandes sont espacées et plafonnées par --max-delta.
                now = time.monotonic()
                if startup_search and now >= next_search_at:
                    sweep = (0, 1, 0, -1, 0, 1, 0, -1)
                    direction = sweep[search_step_index % len(sweep)]
                    search_step_index += 1
                    next_search_at = now + 1.0
                    if direction:
                        search_pan = direction * min(args.max_delta, 120.0)
                        ptz.move(
                            search_pan, 0.0, 0.0, args.max_delta,
                            args.invert_pan, current_invert_tilt,
                        )
                        print("Recherche initiale : balayage horizontal progressif.")
                    if search_step_index >= len(sweep):
                        # Reboucler lentement tant que le visage demeure perdu.
                        search_step_index = 0

                if (
                    not startup_search
                    and args.lost_recenter > 0
                    and not recentered_for_loss
                    and time.monotonic() - lost_since >= args.lost_recenter
                ):
                    recentered_for_loss = True
                    if ptz.center():
                        print("Visage perdu : recentrage.")
                        if stop.wait(1.2):
                            break
                        ptz.refresh_absolute()
                    else:
                        print("Visage perdu : recentrage indisponible.")

                if lost_since is not None:
                    zoom.update_lost(time.monotonic() - lost_since)
                if args.preview and not show_preview(zoomed_frame, None, None):
                    break
                continue

            lost_since = None
            recentered_for_loss = False
            startup_search = False

            # Ne jamais commander plusieurs fois sur la même détection.
            if detected_now:
                x, y, w, h = last_face[:4]
                zoom.update(float(w) / small.shape[1])
                raw_x = (x + w / 2 - DETECT_W / 2) / detection_zoom
                raw_y = (y + h / 2 - DETECT_H / 2) / detection_zoom

                if not has_smooth:
                    smooth_x, smooth_y = raw_x, raw_y
                    has_smooth = True
                else:
                    a = args.smooth
                    smooth_x = a * raw_x + (1 - a) * smooth_x
                    smooth_y = a * raw_y + (1 - a) * smooth_y

                # V4L2 : pan positif vers la droite,
                # tilt positif vers le haut.
                pan = axis_step(smooth_x, args.deadzone[0], args.kp_pan)
                tilt = -axis_step(smooth_y, args.deadzone[1], args.kp_tilt)

                if pan or tilt:
                    ptz.move(
                        pan, tilt, args.motor_interval, args.max_delta,
                        args.invert_pan, current_invert_tilt,
                    )

            if args.preview and not show_preview(
                zoomed_frame, last_face, (smooth_x * detection_zoom,
                                           smooth_y * detection_zoom)
            ):
                break

    finally:
        print("\nArrêt et libération des ressources.")
        if feeder is not None:
            feeder.stop()
        if cam is not None:
            cam.close()
        if ptz is not None:
            ptz.close()


SETTINGS_PATH = os.path.expanduser("~/.config/follow_me/settings.json")
DEFAULT_SETTINGS = {
    "zoom_max": 3.0, "face_target": 0.25, "speed": 1.0,
    "deadzone_x": 16.0, "deadzone_y": 12.0,
    "preview": False, "invert_y": True, "invert_pan": False,
    "zoom_enabled": True, "virtual_video": False, "language": "en",
}

# --- V7 : constantes du service ---
SERVICE_NAME = "follow-me"
SERVICE_DIR = os.path.expanduser("~/.local/share/follow_me")
SERVICE_SCRIPT = os.path.join(SERVICE_DIR, "follow_me_7.py")
SERVICE_SHELL = os.path.join(SERVICE_DIR, "shell.nix")


def load_settings(args):
    values = dict(DEFAULT_SETTINGS)
    values.update({"zoom_max": args.zoom_max,
                   "face_target": args.zoom_target_ratio,
                   "deadzone_x": float(args.deadzone[0]),
                   "deadzone_y": float(args.deadzone[1]),
                   "preview": bool(args.preview),
                   "zoom_enabled": not args.no_auto_zoom,
                   "invert_y": bool(args.invert_tilt),
                   "invert_pan": bool(args.invert_pan),
                   "virtual_video": bool(args.virtual_video)})
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as stream:
            saved = json.load(stream)
        if isinstance(saved, dict):
            for key in values:
                if key in saved and isinstance(saved[key], type(values[key])):
                    values[key] = saved[key]
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError) as exc:
        print(f"[AVERTISSEMENT] Réglages illisibles, valeurs par défaut utilisées ({exc}).",
              file=sys.stderr)
    return values


def save_settings(values):
    os.makedirs(os.path.dirname(SETTINGS_PATH), mode=0o700, exist_ok=True)
    temp_path = SETTINGS_PATH + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as stream:
        json.dump(values, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.chmod(temp_path, 0o600)
    os.replace(temp_path, SETTINGS_PATH)


def check_startup_prerequisites(args, need_tray):
    """Contrôle modules, tray/DBus, v4l2-ctl et capacités YuNet."""
    global cv2, np
    need_virtual_video = bool(getattr(args, "virtual_video", False))
    ensure_nix_environment(need_tray=need_tray, need_gui=bool(args.preview),
                           need_virtual_video=need_virtual_video)
    errors = quick_missing(need_tray=need_tray, need_gui=bool(args.preview),
                           need_virtual_video=need_virtual_video)
    # pystray est facultatif : main() active le mode console si son import échoue.
    errors = [error for error in errors if error != "module Python pystray"]
    pystray_import_failed = False
    if cv2 is None:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            pass
    if np is None:
        try:
            import numpy as _np
            np = _np
        except ImportError:
            pass
    # Importer ici une seconde fois après l'éventuel nix-shell.
    if need_tray:
        for module, label in (("pystray", "pystray"), ("PIL", "Pillow"),
                              ("tkinter", "tkinter")):
            try:
                __import__(module)
            except Exception as exc:
                if module == "pystray":
                    pystray_import_failed = True
                    errors.append(f"module Python pystray : {exc}")
                elif f"module Python {label}" not in errors:
                    errors.append(f"module Python {label} : {exc}")
    if need_tray and not pystray_import_failed:
        available, detail = tray_environment_status()
        print(f"Vérification du tray : {'[OK]' if available else '[ERREUR]'} {detail}.")
        if not available:
            errors.append(detail)
    if pystray_import_failed:
        optional_tray_errors = ("module Python pystray", "module Python Pillow",
                                "module Python tkinter")
        errors = [error for error in errors
                  if not error.startswith(optional_tray_errors)]
    fatal_errors = [error for error in errors
                    if not error.startswith("module Python pystray : ")]
    if fatal_errors:
        errors = fatal_errors
        if is_nixos() and os.environ.get("FOLLOW_ME_IN_NIX_SHELL"):
            advice = "Vérifiez l'installation Nix : opencv4 avec enableGtk3, numpy, pystray, pillow, tk et v4l-utils."
        elif is_nixos():
            advice = "Impossible de démarrer nix-shell automatiquement ; installez opencv4 (enableGtk3), numpy, pystray, pillow, tk et v4l-utils."
        elif distro_id() in ("debian", "ubuntu", "linuxmint", "pop"):
            advice = ("Sur Debian/Ubuntu, installez au besoin : sudo apt install "
                      "python3-opencv python3-numpy v4l-utils ffmpeg python3-pil "
                      "python3-tk python3-pystray. Une version récente d'OpenCV "
                      "avec FaceDetectorYN est nécessaire ; le script n'installe "
                      "pas de paquets automatiquement.")
        else:
            advice = ("Installez OpenCV récent avec YuNet, numpy, v4l-utils et, "
                      "pour le tray, pystray/Pillow/Tk. Les commandes externes "
                      "doivent être accessibles dans PATH.")
        offer_apt_install(errors)
        raise RuntimeError("Prérequis manquants : " + ", ".join(errors) + ". " + advice)
    print("Vérification des prérequis : modules requis et v4l2-ctl disponibles.")


def reset_from_menu(device):
    """Reset matériel équivalent à la commande v4l2-ctl attendue."""
    target = device
    if not target:
        paths = sorted(glob.glob("/dev/v4l/by-id/*-video-index*"))
        if len(paths) != 1:
            raise RuntimeError("Choisissez --device si plusieurs nœuds caméra sont présents.")
        target = paths[0]
    result = subprocess.run([require_command("v4l2-ctl"), "-d", target,
                             "--set-ctrl=pan_reset=1,tilt_reset=1"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=10)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Le reset pan/tilt a échoué.")
    print("Reset pan/tilt envoyé.")


class TrayApplication:
    """Tkinter reste dans le thread principal; suivi et icône ont leurs threads."""
    def __init__(self, args, settings):
        import tkinter as tk
        from tkinter import messagebox
        import pystray
        from PIL import Image, ImageDraw, ImageTk
        self.tk, self.pystray = tk, pystray
        self.messagebox = messagebox
        self.Image, self.ImageDraw, self.ImageTk = Image, ImageDraw, ImageTk
        self.args = args
        self.shared = {"lock": threading.Lock(), "values": settings}
        self.commands = queue.Queue()
        self.worker = None
        self.worker_stop = None
        self.manual_window = None
        self.manual_buttons = []
        self.manual_status = None
        self.manual_thread = None
        self.manual_stop = None
        self.manual_commands = None
        self.manual_status_queue = queue.Queue()
        self._manual_pending = False
        self._tracking_pending = False
        self._last_tracking_state = False
        self.closed = threading.Event()
        self.root = tk.Tk()
        self.root.title(tr("settings_title"))
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.hide_settings)
        self.root.withdraw()
        self.preview_window = tk.Toplevel(self.root)
        self.preview_window.protocol("WM_DELETE_WINDOW", self._hide_preview)
        self.preview_window.title(tr("preview_title"))
        self.preview_label = tk.Label(self.preview_window, text=tr("preview_waiting"))
        self.preview_label.pack(padx=8, pady=8)
        self.preview_photo = None
        self.preview_window.withdraw()
        self.vars = {}
        self.icon = None
        self._build_settings()

    def _build_settings(self):
        tk = self.tk
        frame = tk.Frame(self.root, padx=18, pady=14)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text=tr("settings_heading"), font=("Sans", 14, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        specs = [("zoom_max", "max_zoom", 1.0, 8.0, 0.1),
                 ("face_target", "face_target", 0.10, 0.80, 0.01),
                 ("speed", "tracking_speed", 0.2, 3.0, 0.05),
                 ("deadzone_x", "deadzone_x", 0, 100, 1),
                 ("deadzone_y", "deadzone_y", 0, 100, 1)]
        with self.shared["lock"]:
            values = dict(self.shared["values"])
        for row, (key, label_key, low, high, resolution) in enumerate(specs, 1):
            tk.Label(frame, text=tr(label_key)).grid(row=row, column=0, sticky="w", pady=3)
            var = tk.DoubleVar(value=values[key])
            self.vars[key] = var
            tk.Scale(frame, variable=var, from_=low, to=high, resolution=resolution,
                     orient="horizontal", length=250, showvalue=True).grid(
                         row=row, column=1, sticky="ew")
        checkbox_specs = (("preview", "preview_video"), ("invert_y", "invert_tilt"),
                          ("invert_pan", "invert_pan"), ("virtual_video", "virtual_video"))
        for row, (key, label_key) in enumerate(checkbox_specs, 6):
            var = tk.BooleanVar(value=values[key]); self.vars[key] = var
            tk.Checkbutton(frame, text=tr(label_key), variable=var).grid(
                row=row, column=0, columnspan=2, sticky="w", pady=3)
        tk.Label(frame, text=tr("language")).grid(row=10, column=0, sticky="w", pady=3)
        lang_var = tk.StringVar(value=values.get("language", "en")); self.vars["language"] = lang_var
        language_menu = tk.OptionMenu(frame, lang_var, "en", "fr")
        language_menu["menu"].delete(0, "end")
        language_menu["menu"].add_command(label=tr("english_language"), command=lambda: lang_var.set("en"))
        language_menu["menu"].add_command(label=tr("french_language"), command=lambda: lang_var.set("fr"))
        language_menu.grid(row=10, column=1, sticky="w")
        tk.Button(frame, text=tr("save"), command=self.apply_settings).grid(
            row=11, column=0, sticky="w", pady=(10, 0))
        tk.Button(frame, text=tr("close"), command=self.hide_settings).grid(
            row=11, column=1, sticky="e", pady=(10, 0))
        frame.columnconfigure(1, weight=1)

    def _read_setting_var(self, var):
        """Read a Tk variable, accepting decimal commas for DoubleVar values."""
        try:
            return var.get()
        except self.tk.TclError:
            if not isinstance(var, self.tk.DoubleVar):
                raise

            # DoubleVar.get() asks Tcl to parse its value as a float and raises
            # before Python can normalize a locale-style decimal comma.
            try:
                raw_value = self.root.globalgetvar(var._name)
            except (self.tk.TclError, AttributeError):
                raw_value = var._tk.globalgetvar(var._name)
            return float(str(raw_value).strip().replace(",", "."))

    def apply_settings(self):
        with self.shared["lock"]:
            self.shared["values"].update(
                {key: self._read_setting_var(var) for key, var in self.vars.items()})
            settings = dict(self.shared["values"])
        try:
            save_settings(settings)
            if settings.get("language", "en") != CURRENT_LANGUAGE:
                self.messagebox.showinfo(tr("app_name"), tr("language_restart"))
            if self.icon is not None:
                self.icon.menu = self._build_menu()
            print(tr("settings_saved"))
        except OSError as exc:
            self.messagebox.showerror(tr("app_name"), tr("save_error", error=exc))

    def show_settings(self):
        self.root.deiconify(); self.root.lift()

    def hide_settings(self):
        self.root.withdraw()

    def _tracking_is_active(self):
        """Retourne l'état réel du suivi, d'après la vie du thread worker."""
        return self.worker is not None and self.worker.is_alive()

    def _refresh_tracking_menu_if_needed(self):
        """Rafraîchit le libellé du menu quand l'état réel change."""
        active = self._tracking_is_active()
        if active != self._last_tracking_state:
            self._last_tracking_state = active
            if self.icon is not None:
                self.icon.menu = self._build_menu()

    def _toggle_tracking(self):
        """Bascule le suivi selon l'état effectif du thread de suivi."""
        if self._tracking_is_active():
            self._stop_tracking()
        else:
            self._start_tracking()

    def _start_tracking(self):
        if self._tracking_is_active():
            return
        if self.manual_thread is not None and self.manual_thread.is_alive():
            self._tracking_pending = True
            self._stop_manual(close_window=True)
            return
        self.worker_stop = threading.Event()
        self.worker = threading.Thread(target=self._tracking_worker,
                                       name="suivi-visage", daemon=True)
        self.worker.start()
        print("Démarrage du suivi demandé.")

    def _tracking_worker(self):
        args = argparse.Namespace(**vars(self.args))
        try:
            run_tracking(args, self.shared, self.worker_stop)
        except Exception as exc:
            print(f"Erreur du suivi : {exc}", file=sys.stderr)

    def _stop_tracking(self):
        if self.worker_stop is not None:
            self.worker_stop.set()
            print("Arrêt du suivi demandé.")

    def _toggle_zoom(self):
        with self.shared["lock"]:
            self.shared["values"]["zoom_enabled"] = not self.shared["values"]["zoom_enabled"]
            enabled = self.shared["values"]["zoom_enabled"]
            snapshot = dict(self.shared["values"])
        save_settings(snapshot)
        if self.icon is not None:
            self.icon.menu = self._build_menu()
        print(f"Zoom automatique {'activé' if enabled else 'désactivé'}.")

    def _hide_preview(self):
        """Masque la fenêtre d'aperçu sans la détruire."""
        self.preview_window.withdraw()
        with self.shared["lock"]:
            self.shared["values"]["preview"] = False
        save_settings(dict(self.shared["values"]))
        if self.icon is not None:
            self.icon.menu = self._build_menu()

    def _toggle_virtual_video(self):
        """Active/désactive le partage vers v4l2loopback."""
        with self.shared["lock"]:
            values = self.shared["values"]
            values["virtual_video"] = not values.get("virtual_video", False)
            enabled = values["virtual_video"]
            snapshot = dict(values)
        save_settings(snapshot)
        if self.icon is not None:
            self.icon.menu = self._build_menu()
        print(f"VirtualVideo {'activée' if enabled else 'désactivée'}.")

    def _toggle_preview(self):
        """Active/désactive l'aperçu vidéo."""
        with self.shared["lock"]:
            self.shared["values"]["preview"] = not self.shared["values"]["preview"]
            enabled = self.shared["values"]["preview"]
            snapshot = dict(self.shared["values"])
        save_settings(snapshot)
        if self.icon is not None:
            self.icon.menu = self._build_menu()
        if not enabled:
            try:
                if self.preview_label.winfo_exists():
                    self.preview_window.withdraw()
            except self.tk.TclError:
                pass
        print(f"Aperçu vidéo {'activé' if enabled else 'désactivé'}.")

    def _rebuild_preview(self):
        """Recrée la fenêtre d'aperçu si elle a été détruite."""
        self.preview_window = self.tk.Toplevel(self.root)
        self.preview_window.title(tr("preview_title"))
        self.preview_window.protocol("WM_DELETE_WINDOW", self._hide_preview)
        self.preview_label = self.tk.Label(self.preview_window, text=tr("preview_waiting"))
        self.preview_label.pack(padx=8, pady=8)
        self.preview_photo = None

    def _reset(self):
        try:
            reset_from_menu(self.args.device)
        except Exception as exc:
            print(f"Reset impossible : {exc}", file=sys.stderr)

    def _show_manual(self):
        """Ouvre le pad; le PTZ manuel démarre après libération du suivi."""
        if self.manual_window is not None:
            try:
                if self.manual_window.winfo_exists():
                    self.manual_window.deiconify()
                    self.manual_window.lift()
                    return
            except self.tk.TclError:
                pass
        window = self.tk.Toplevel(self.root)
        self.manual_window = window
        window.title(tr("manual_title"))
        window.resizable(False, False)
        window.protocol("WM_DELETE_WINDOW", self._close_manual)
        frame = self.tk.Frame(window, padx=14, pady=12)
        frame.pack()
        self.manual_status = self.tk.Label(frame, text=tr("camera_connecting"), width=34)
        self.manual_status.grid(row=0, column=0, columnspan=3, pady=(0, 5))
        self.tk.Label(frame, text=tr("hold_direction")).grid(
            row=1, column=0, columnspan=3, pady=(0, 4))
        self.manual_buttons = []
        for row, col, text, pan, tilt in (
            (2, 1, tr("up"), 0, 1), (3, 0, tr("left"), -1, 0),
            (3, 1, tr("stop"), 0, 0), (3, 2, tr("right"), 1, 0),
            (4, 1, tr("down"), 0, -1),
        ):
            button = self.tk.Button(frame, text=text, width=8, height=2, state="disabled")
            button.grid(row=row, column=col, padx=3, pady=3)
            if pan or tilt:
                button.bind("<ButtonPress-1>",
                            lambda _event, p=pan, t=tilt: self._manual_press(p, t))
                button.bind("<ButtonRelease-1>", lambda _event: self._manual_press(0, 0))
                button.bind("<Leave>", lambda _event: self._manual_press(0, 0))
            else:
                button.configure(command=lambda: self._manual_press(0, 0))
            self.manual_buttons.append(button)
        center = self.tk.Button(frame, text=tr("center"), state="disabled",
                                command=lambda: self._manual_press(0, 0, center=True))
        center.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        self.manual_buttons.append(center)
        self._manual_pending = True
        self._tracking_pending = False
        if self._tracking_is_active():
            self._stop_tracking()
            self.manual_status.configure(text=tr("stopping_tracking"))
        self._advance_modes()

    def _manual_press(self, pan, tilt, center=False):
        if self.manual_commands is not None:
            self.manual_commands.put(("center" if center else "move", pan, tilt))

    def _set_manual_controls(self, enabled):
        for button in self.manual_buttons:
            try:
                button.configure(state="normal" if enabled else "disabled")
            except self.tk.TclError:
                pass

    def _close_manual(self):
        self._manual_pending = False
        self._stop_manual(close_window=True)

    def _stop_manual(self, close_window=False):
        if self.manual_stop is not None:
            self.manual_stop.set()
        if self.manual_commands is not None:
            self.manual_commands.put(("stop", 0, 0))
        if close_window and self.manual_window is not None:
            try:
                self.manual_window.destroy()
            except self.tk.TclError:
                pass
            self.manual_window = None

    def _manual_worker(self):
        ptz = None
        cam = None
        try:
            ptz = select_camera(self.args.device)
            # Le mode manuel arrêtait le worker de suivi (et donc sa capture)
            # avant d'ouvrir PTZ. Garder un FreshCamera dédié ici : le thread
            # Tk du tray continue de lire PREVIEW_QUEUE sans accéder à Tk depuis
            # ce worker, tandis que le descripteur PTZ reste réservé aux commandes.
            cam = FreshCamera(ptz.path, self.args.width, self.args.height, self.args.fps)
            self.manual_status_queue.put(("ready", tr("camera_connected", mode=ptz.mode)))
            direction = (0, 0)
            last_move = 0.0
            interval = max(0.08, float(self.args.motor_interval))
            step = min(50.0, float(self.args.max_delta))
            last_seq = -1
            last_frame_at = time.monotonic()
            while not self.manual_stop.is_set():
                try:
                    command, pan, tilt = self.manual_commands.get(timeout=0.03)
                    if command == "stop":
                        direction = (0, 0)
                    elif command == "move":
                        direction = (pan, tilt)
                    elif command == "center":
                        direction = (0, 0)
                        if not ptz.center():
                            self.manual_status_queue.put(("message", tr("center_unavailable")))
                        else:
                            self.manual_status_queue.put(("message", tr("camera_centered")))
                except queue.Empty:
                    pass

                frame, seq = cam.latest(last_seq)
                if frame is not None:
                    last_seq = seq
                    last_frame_at = time.monotonic()
                    with self.shared["lock"]:
                        preview_enabled = bool(self.shared["values"].get("preview", False))
                    if preview_enabled:
                        show_preview(frame, None, None)
                elif time.monotonic() - last_frame_at > self.args.frame_timeout:
                    raise RuntimeError(
                        tr("no_new_frame")
                    )

                now = time.monotonic()
                if direction != (0, 0) and now - last_move >= interval:
                    with self.shared["lock"]:
                        invert_tilt = bool(self.shared["values"].get("invert_y", True))
                        invert_pan = bool(self.shared["values"].get("invert_pan", False))
                    try:
                        ptz.move(direction[0] * step, direction[1] * step,
                                 self.args.motor_interval, self.args.max_delta,
                                 invert_pan, invert_tilt)
                    except OSError as exc:
                        self.manual_status_queue.put(("error", tr("move_error", error=exc)))
                        break
                    last_move = now
        except Exception as exc:
            self.manual_status_queue.put(("error", tr("manual_error", error=exc)))
        finally:
            try:
                if cam is not None:
                    cam.close()
            finally:
                if ptz is not None:
                    ptz.close()
                self.manual_status_queue.put(("stopped", tr("manual_stopped")))

    def _advance_modes(self):
        if self._manual_pending and not self._tracking_is_active() and not (
                self.manual_thread is not None and self.manual_thread.is_alive()):
            self._manual_pending = False
            self.manual_stop = threading.Event()
            self.manual_commands = queue.Queue()
            self.manual_thread = threading.Thread(target=self._manual_worker,
                                                  name="pilotage-manuel", daemon=True)
            self.manual_thread.start()
        if self._tracking_pending and not (
                self.manual_thread is not None and self.manual_thread.is_alive()):
            self._tracking_pending = False
            self._start_tracking()

    def _poll_manual_status(self):
        while True:
            try:
                kind, text = self.manual_status_queue.get_nowait()
            except queue.Empty:
                break
            if self.manual_status is not None:
                try:
                    self.manual_status.configure(text=text)
                except self.tk.TclError:
                    pass
            if kind == "ready":
                self._set_manual_controls(True)
            elif kind in ("error", "stopped"):
                self._set_manual_controls(False)
            if kind == "error":
                print(text, file=sys.stderr)

    def _icon_image(self):
        image = self.Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = self.ImageDraw.Draw(image)
        draw.ellipse((5, 5, 59, 59), fill=(37, 99, 235, 255))
        draw.ellipse((20, 16, 44, 40), outline="white", width=4)
        draw.arc((13, 24, 51, 54), 15, 165, fill="white", width=4)
        return image

    def _build_menu(self):
        """Construit le menu du tray avec l'état actuel des options."""
        pystray = self.pystray
        with self.shared["lock"]:
            zoom_enabled = bool(self.shared["values"]["zoom_enabled"])
            preview_enabled = bool(self.shared["values"]["preview"])
            virtual_enabled = bool(self.shared["values"].get("virtual_video", False))
        zoom_text = tr("zoom_on") if zoom_enabled else tr("zoom_off")
        preview_text = tr("preview_on") if preview_enabled else tr("preview_off")
        virtual_text = tr("virtual_on") if virtual_enabled else tr("virtual_off")
        tracking_text = tr("stop_tracking") if self._tracking_is_active() else tr("start_tracking")
        return pystray.Menu(
            pystray.MenuItem(tracking_text, lambda icon, item: self.commands.put("toggle_tracking")),
            pystray.MenuItem(zoom_text, lambda icon, item: self.commands.put("zoom")),
            pystray.MenuItem(preview_text, lambda icon, item: self.commands.put("preview")),
            pystray.MenuItem(virtual_text, lambda icon, item: self.commands.put("virtual_video")),
            pystray.MenuItem(tr("manual_menu"), lambda icon, item: self.commands.put("manual")),
            pystray.MenuItem(tr("reset_menu"), lambda icon, item: self.commands.put("reset")),
            pystray.MenuItem(tr("settings_menu"), lambda icon, item: self.commands.put("settings")),
            pystray.MenuItem(tr("quit"), lambda icon, item: self.commands.put("quit")),
        )

    def _run_icon(self):
        self.icon = self.pystray.Icon("follow_me", self._icon_image(), tr("app_name"), menu=self._build_menu())
        try:
            self.icon.run()
        except Exception as exc:
            print(f"Erreur du tray : {exc}", file=sys.stderr)
            self.commands.put("tray_error")

    def _poll_commands(self):
        if getattr(self, "_closing", False):
            return
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                break
            if command == "toggle_tracking":
                self._toggle_tracking()
            elif command == "zoom":
                self._toggle_zoom()
            elif command == "preview":
                self._toggle_preview()
            elif command == "virtual_video":
                self._toggle_virtual_video()
            elif command == "reset":
                self._reset()
            elif command == "manual":
                self._show_manual()
            elif command == "settings":
                self.show_settings()
            elif command in ("quit", "tray_error"):
                self.shutdown()
                return
            if getattr(self, "_closing", False):
                return

        # Aussi détecte les terminaisons autonomes (erreur, caméra débranchée,
        # sortie de la boucle) et met à jour le menu après un démarrage/arrêt.
        self._advance_modes()
        self._poll_manual_status()
        self._refresh_tracking_menu_if_needed()

        try:
            frame = PREVIEW_QUEUE.get_nowait()
        except queue.Empty:
            frame = None

        # Respecter le réglage live de l'aperçu
        with self.shared["lock"]:
            preview_enabled = bool(self.shared["values"]["preview"])

        try:
            if frame is not None and preview_enabled:
                # Recréer la fenêtre si elle a été détruite
                if not self.preview_label.winfo_exists():
                    self._rebuild_preview()
                image = self.Image.fromarray(frame[:, :, ::-1])
                self.preview_photo = self.ImageTk.PhotoImage(image=image)
                self.preview_label.configure(image=self.preview_photo, text="")
                if not self.preview_window.winfo_viewable():
                    self.preview_window.deiconify()
            elif not preview_enabled:
                # Masquer l'aperçu si désactivé
                if self.preview_label.winfo_exists() and self.preview_window.winfo_viewable():
                    self.preview_window.withdraw()
        except self.tk.TclError:
            return

        self._after_id = self.root.after(150, self._poll_commands)

    def shutdown(self):
        if self.closed.is_set():
            return
        self.closed.set()
        self._closing = True

        if getattr(self, "_after_id", None):
            try:
                self.root.after_cancel(self._after_id)
            except self.tk.TclError:
                pass
            self._after_id = None

        self._stop_tracking()
        self._manual_pending = False
        self._stop_manual(close_window=True)

        if self.icon is not None:
            try:
                self.icon.stop()
            except Exception:
                pass

        try:
            self.root.quit()
            self.root.destroy()
        except self.tk.TclError:
            pass

    def run(self):
        self.icon_thread = threading.Thread(target=self._run_icon, name="icône-tray", daemon=True)
        self.icon_thread.start()
        self.root.after(150, self._poll_commands)
        self.root.mainloop()
        self._stop_tracking()
        if self.worker is not None:
            self.worker.join(timeout=3)
        if self.manual_thread is not None:
            self.manual_thread.join(timeout=3)
        self.closed.set()


def run_console_preview(args, stop=None):
    """Affiche l’aperçu Tk dans le mode terminal sans tray."""
    import tkinter as tk
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title(tr("preview_title"))
    label = tk.Label(root, text=tr("preview_waiting"))
    label.pack(padx=8, pady=8)
    stop = stop if stop is not None else threading.Event()
    worker_errors = []
    photo = {"current": None}

    def tracking_worker():
        try:
            run_tracking(args, stop=stop)
        except Exception as exc:
            worker_errors.append(exc)
            stop.set()

    worker = threading.Thread(target=tracking_worker, name="suivi-visage", daemon=True)
    worker.start()

    def request_stop(_event=None):
        stop.set()

    root.protocol("WM_DELETE_WINDOW", request_stop)
    root.bind("<KeyPress-q>", request_stop)

    def poll_preview():
        try:
            frame = PREVIEW_QUEUE.get_nowait()
        except queue.Empty:
            frame = None
        if frame is not None:
            image = Image.fromarray(frame[:, :, ::-1])
            photo["current"] = ImageTk.PhotoImage(image=image)
            label.configure(image=photo["current"], text="")
        if worker.is_alive():
            root.after(30, poll_preview)
        else:
            root.quit()

    root.after(30, poll_preview)
    root.mainloop()
    stop.set()
    worker.join(timeout=3.0)
    root.destroy()
    if worker_errors:
        raise worker_errors[0]
def build_service_unit(args):
    """Construit le lancement systemd adapté à NixOS ou Debian."""
    extra = [arg for arg in sys.argv[1:]
             if arg not in ("--install-service", "--uninstall-service")]
    if is_nixos():
        inner = " ".join(shlex.quote(arg) for arg in [SERVICE_SCRIPT, *extra])
        nix_shell = require_command("nix-shell")
        exec_start = (
            f"{shlex.quote(nix_shell)} {shlex.quote(SERVICE_SHELL)} --run "
            f"{shlex.quote('python3 ' + inner)}"
        )
    else:
        # Debian/systemd : le même interpréteur de l'installation est utilisé,
        # sans présumer que python soit dans /usr/bin ni requérir nix-shell.
        python = sys.executable or shutil.which("python3")
        if not python:
            raise RuntimeError("Interpréteur Python 3 introuvable.")
        exec_start = " ".join(shlex.quote(arg) for arg in [python, SERVICE_SCRIPT, *extra])
    return f"""[Unit]
Description=Suivi de visage PTZ (follow_me)
PartOf=graphical-session.target
After=graphical-session.target

[Service]
Type=simple
ExecStart={exec_start}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=graphical-session.target
"""


def install_service(args):
    """Copie le script et active une unité systemd utilisateur."""
    require_command("systemctl")
    if is_nixos():
        require_command("nix-shell")
    unit_dir = os.path.expanduser("~/.config/systemd/user")
    os.makedirs(SERVICE_DIR, mode=0o700, exist_ok=True)
    os.makedirs(unit_dir, exist_ok=True)

    shutil.copy2(os.path.abspath(__file__), SERVICE_SCRIPT)
    language_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_driver.lng")
    if os.path.isfile(language_src):
        shutil.copy2(language_src, os.path.join(SERVICE_DIR, "camera_driver.lng"))
    model_src = DEFAULT_MODEL
    if os.path.isfile(model_src):
        shutil.copy2(model_src, os.path.join(SERVICE_DIR, MODEL_NAME))
        print(f"Modèle copié : {os.path.join(SERVICE_DIR, MODEL_NAME)}")
    else:
        print("[AVERTISSEMENT] Modèle YuNet introuvable à côté du script ; le suivi "
              "ne démarrera pas. Copie-le dans " + SERVICE_DIR, file=sys.stderr)
    os.chmod(SERVICE_SCRIPT, 0o700)
    if is_nixos():
        with open(SERVICE_SHELL, "w", encoding="utf-8") as stream:
            stream.write(SHELL_NIX)
        os.chmod(SERVICE_SHELL, 0o600)

    unit_path = os.path.join(unit_dir, f"{SERVICE_NAME}.service")
    with open(unit_path, "w", encoding="utf-8") as stream:
        stream.write(build_service_unit(args))
    print(f"Unité écrite : {unit_path}")
    print(f"Script copié : {SERVICE_SCRIPT}")

    subprocess.run([require_command("systemctl"), "--user", "daemon-reload"], check=True)
    subprocess.run([require_command("systemctl"), "--user", "enable", "--now", SERVICE_NAME], check=True)
    print(f"Service activé et démarré : systemctl --user status {SERVICE_NAME}")
    print("Note : le service démarre avec la session graphique ; "
          "pour les réglages, lance le script normalement (tray) ou édite "
          "~/.config/follow_me/settings.json.")


def uninstall_service():
    """Arrête et supprime le service utilisateur."""
    systemctl = shutil.which("systemctl")
    if systemctl is None:
        print("systemctl introuvable dans PATH ; l'unité ne peut pas être arrêtée.",
              file=sys.stderr)
        return
    subprocess.run([systemctl, "--user", "disable", "--now", SERVICE_NAME],
                   check=False)
    unit_path = os.path.expanduser(f"~/.config/systemd/user/{SERVICE_NAME}.service")
    try:
        os.remove(unit_path)
    except FileNotFoundError:
        pass
    subprocess.run([systemctl, "--user", "daemon-reload"], check=False)
    subprocess.run([systemctl, "--user", "reset-failed"], check=False)
    print("Service follow-me supprimé (les fichiers dans ~/.local/share/follow_me "
          "et les réglages sont conservés).")

def close_previous_instances():
    """Ferme les autres scripts follow_me du même utilisateur."""
    current_pid = os.getpid()
    current_uid = os.getuid()
    username = getpass.getuser()

    def is_follow_me(pid):
        try:
            with open(f"/proc/{pid}/status", encoding="utf-8") as stream:
                status = stream.read()
            uid_line = next(line for line in status.splitlines() if line.startswith("Uid:"))
            if int(uid_line.split()[1]) != current_uid:
                return False
            with open(f"/proc/{pid}/cmdline", "rb") as stream:
                argv = [part.decode(errors="replace") for part in stream.read().split(b"\0") if part]
        except (OSError, StopIteration, ValueError):
            return False
        executable = os.path.basename(argv[0]) if argv else ""
        if executable.startswith("follow_me") and executable.endswith(".py"):
            return True
        for index, argument in enumerate(argv[1:], 1):
            name = os.path.basename(argument)
            if argument == "-m" and index + 1 < len(argv):
                module = argv[index + 1].split(".", 1)[0]
                if module == "follow_me":
                    return True
            if name == "follow_me" or (name.startswith("follow_me") and name.endswith(".py")):
                return True
        return False

    previous = []
    try:
        pids = (int(entry) for entry in os.listdir("/proc") if entry.isdigit())
        for pid in pids:
            if pid != current_pid and is_follow_me(pid):
                previous.append(pid)
    except OSError as exc:
        print(f"[AVERTISSEMENT] Lecture de /proc impossible : {exc}", file=sys.stderr)
        return

    if not previous:
        return
    print(f"Fermeture de {len(previous)} instance(s) follow_me précédente(s) de {username}.")
    for pid in previous:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            continue

    deadline = time.monotonic() + 2.0
    pending = set(previous)
    while pending and time.monotonic() < deadline:
        pending = {pid for pid in pending if is_follow_me(pid)}
        if pending:
            time.sleep(0.1)
    for pid in pending:
        if is_follow_me(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

def main():
    args = parse_args()
    if args.uninstall_service:
        uninstall_service()
        return
    if args.install_service:
        install_service(args)
        return
    close_previous_instances()
    need_tray = not args.no_tray
    settings = load_settings(args) if need_tray else None
    global CURRENT_LANGUAGE
    if settings is not None:
        CURRENT_LANGUAGE = settings.get("language", "en") if settings.get("language") in ("en", "fr") else "en"
    if need_tray and settings["preview"]:
        args.preview = True
    check_startup_prerequisites(args, need_tray)
    if need_tray:
        try:
            import pystray  # noqa: F401
        except Exception as exc:
            print(f"Avertissement : impossible de charger pystray ({exc}) ; poursuite sans icône de tray.",
                  file=sys.stderr)
            args.no_tray = True
            need_tray = False
    if args.no_tray:
        stop = threading.Event()
        def request_stop(_signum, _frame): stop.set()
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        # Mode console historique : les options CLI gardent la priorité et la
        # boucle de suivi s'exécute dans le thread principal.
        if args.preview:
            run_console_preview(args, stop=stop)
        else:
            run_tracking(args, stop=stop)
        return
    app = TrayApplication(args, settings)
    try:
        app.run()
    except KeyboardInterrupt:
        app.shutdown()


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        if cv2 is not None and isinstance(exc, cv2.error):
            print(f"Erreur : {exc}", file=sys.stderr)
            sys.exit(1)
        raise
