#pragma once

#include <filesystem>
#include <vector>
#include <string>

enum class RAMSize {
    eNone,
    eKB2,
    eKB8,
    eKB32,
    eKB128,
    eKB64,
};

enum class ROMSize {
    eKB32,
    eKB64,
    eKB128,
    eKB256,
    eKB512,
    eMB1,
    eMB2,
    eMB4,
    eMB1p1,
    eMB1p2,
    eMB1p5,
};

enum class CartridgeType {
    eROMOnly,
    eMBC1,
    eMBC2,
    eMBC3,
    eMBC4,
    eMBC5
};

class CartridgeInfo {
public:
    CartridgeInfo(const std::vector<uint8_t>&);

    std::string title;

    CartridgeType type;
    bool destination; // false - Japanese, true - non-Japanese
    ROMSize rom_size;
    RAMSize ram_size;
    std::string license_code;
    uint8_t version;

    uint16_t header_checksum;
    uint16_t global_checksum;

    bool supports_cgb;
    bool supports_sgb;
};

class Cartridge {
public:
    Cartridge(const char*);

    uint8_t read(const uint16_t& address) const;
    const std::vector<uint8_t>& get_rom() const;
private:
    std::vector<uint8_t> rom;
    CartridgeInfo info;
};

