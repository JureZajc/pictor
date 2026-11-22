from pictor import hardware
from pictor.hardware import Gpu, compute_budget, guess_vendor, looks_discrete, parse_nvidia_smi


def test_apple_silicon_uses_a_share_of_unified_memory():
    accelerator, budget, note = compute_budget("macOS", "arm64", 24.0, [])
    assert accelerator == "apple-silicon"
    assert budget == 16.8
    assert "24 GB" in note


def test_intel_mac_falls_back_to_cpu():
    accelerator, budget, _ = compute_budget("macOS", "x86_64", 32.0, [])
    assert accelerator == "cpu"
    assert budget == 8.0


def test_nvidia_card_sets_the_budget_from_its_vram():
    gpus = [Gpu("NVIDIA GeForce RTX 4090", "nvidia", 24.0, True)]
    accelerator, budget, note = compute_budget("Windows", "amd64", 64.0, gpus)
    assert accelerator == "nvidia"
    assert budget == 22.8
    assert "NVIDIA" in note


def test_several_cards_of_one_vendor_add_up():
    gpus = [Gpu("RTX 3090", "nvidia", 24.0, True), Gpu("RTX 3090", "nvidia", 24.0, True)]
    accelerator, budget, note = compute_budget("Linux", "x86_64", 128.0, gpus)
    assert (accelerator, budget) == ("nvidia", 45.6)
    assert "2 NVIDIA GPUs" in note


def test_a_small_gpu_is_still_preferred_over_a_larger_cpu_budget():
    # 6 GB of VRAM gives a smaller budget than the CPU cap, but runs far faster.
    gpus = [Gpu("GTX 1660", "nvidia", 6.0, True)]
    assert compute_budget("Linux", "x86_64", 64.0, gpus)[:2] == ("nvidia", 5.7)


def test_display_adapters_do_not_count_as_gpus():
    gpus = [Gpu("GT 710", "nvidia", 2.0, False), Gpu("Intel UHD Graphics", "intel", None, False)]
    assert compute_budget("Windows", "amd64", 16.0, gpus)[:2] == ("cpu", 8.0)


def test_cpu_budget_is_half_the_ram_up_to_a_cap():
    assert compute_budget("Linux", "x86_64", 8.0, [])[1] == 4.0
    assert compute_budget("Linux", "x86_64", 128.0, [])[1] == 8.0


def test_amd_and_intel_arc_cards_are_used():
    amd = [Gpu("AMD Radeon RX 7900 XT", "amd", 20.0, True)]
    assert compute_budget("Linux", "x86_64", 32.0, amd)[:2] == ("amd", 19.0)
    arc = [Gpu("Intel Arc A770", "intel", 16.0, True)]
    assert compute_budget("Windows", "amd64", 32.0, arc)[:2] == ("intel", 15.2)


def test_parse_nvidia_smi_reads_names_and_mebibytes():
    text = "NVIDIA GeForce RTX 4090, 24564\nNVIDIA GeForce GT 710, 2048\n\n"
    cards = parse_nvidia_smi(text)
    assert [c.name for c in cards] == ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce GT 710"]
    assert cards[0].vram_gb == 24.0 and cards[0].discrete
    assert cards[1].vram_gb == 2.0 and not cards[1].discrete


def test_parse_nvidia_smi_survives_unreadable_memory():
    (card,) = parse_nvidia_smi("Tesla T4, [N/A]")
    assert card.vram_gb is None and not card.discrete


def test_guess_vendor():
    assert guess_vendor("NVIDIA GeForce RTX 5080") == "nvidia"
    assert guess_vendor("AMD Radeon(TM) Graphics") == "amd"
    assert guess_vendor("Intel(R) Arc(TM) A770 Graphics") == "intel"
    assert guess_vendor("Apple M5 Pro GPU") == "apple"
    assert guess_vendor("Microsoft Basic Display Adapter") == "other"


def test_looks_discrete():
    assert looks_discrete("Radeon RX 6800", "amd", 16.0)
    assert not looks_discrete("AMD Radeon(TM) Graphics", "amd", 0.5)
    assert not looks_discrete("Intel(R) UHD Graphics 770", "intel", 8.0)
    assert looks_discrete("Intel(R) Arc(TM) A770 Graphics", "intel", 16.0)
    assert not looks_discrete("Mystery card", "other", 24.0)


def test_detect_describes_this_machine_without_raising():
    hardware.detect.cache_clear()
    machine = hardware.detect()
    assert machine.ram_gb > 0
    assert machine.cores >= 1
    assert machine.budget_gb > 0
    assert machine.accelerator in {"apple-silicon", "nvidia", "amd", "intel", "cpu"}
    assert machine.cpu and machine.summary
    assert machine.to_dict()["budget_gb"] == machine.budget_gb
