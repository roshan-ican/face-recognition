// Command gencert creates a self-signed TLS certificate for local development.
//
// Run it with:  go run ./tools/gencert
//
// It writes certs/cert.pem (the certificate, public) and certs/key.pem
// (the private key, secret). Both are read by main.go at startup.
package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"log"
	"math/big"
	"net"
	"os"
	"time"
)

// Every name/IP we might type into the browser's address bar must appear here.
// Add more entries if your LAN IP changes.
var (
	hostNames = []string{"localhost"}
	hostIPs   = []string{"127.0.0.1", "::1", "10.252.176.230"}
)

func main() {
	// 1. Generate a private key. This is the secret half of the pair.
	//    P-256 is an elliptic curve; it's smaller and faster than RSA and
	//    every modern browser supports it.
	privateKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		log.Fatal("generating key:", err)
	}

	// 2. Certificates need a unique serial number. For a real CA this
	//    matters a lot; for us a large random number is plenty.
	serialLimit := new(big.Int).Lsh(big.NewInt(1), 128) // 2^128
	serialNumber, err := rand.Int(rand.Reader, serialLimit)
	if err != nil {
		log.Fatal("generating serial number:", err)
	}

	// 3. Describe the certificate we want.
	template := x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			Organization: []string{"face-recognition dev"},
		},
		NotBefore: time.Now().Add(-time.Hour), // guard against clock skew
		NotAfter:  time.Now().AddDate(1, 0, 0),

		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  true, // self-signed means we are our own CA

		// THE important part: Subject Alternative Names.
		DNSNames:    hostNames,
		IPAddresses: parseIPs(hostIPs),
	}

	// 4. Sign it. Passing `template` twice is what makes it *self*-signed:
	//    the certificate and its issuer are the same document.
	derBytes, err := x509.CreateCertificate(
		rand.Reader,
		&template,             // the certificate to create
		&template,             // the issuer (us)
		&privateKey.PublicKey, // the public key to embed
		privateKey,            // the key that signs
	)
	if err != nil {
		log.Fatal("creating certificate:", err)
	}

	// 5. Write both files in PEM format (base64 wrapped in BEGIN/END lines).
	writePEM("certs/cert.pem", "CERTIFICATE", derBytes, 0o644)

	keyBytes, err := x509.MarshalECPrivateKey(privateKey)
	if err != nil {
		log.Fatal("marshalling key:", err)
	}
	writePEM("certs/key.pem", "EC PRIVATE KEY", keyBytes, 0o600)

	fmt.Println("Wrote certs/cert.pem and certs/key.pem")
	fmt.Println("Valid for:", hostNames, hostIPs)
	fmt.Println("Expires:", template.NotAfter.Format(time.RFC1123))
}

// parseIPs converts strings like "10.252.176.230" into net.IP values,
// failing loudly if one is malformed.
func parseIPs(raw []string) []net.IP {
	ips := make([]net.IP, 0, len(raw))
	for _, s := range raw {
		ip := net.ParseIP(s)
		if ip == nil {
			log.Fatalf("not a valid IP address: %q", s)
		}
		ips = append(ips, ip)
	}
	return ips
}

func writePEM(path, blockType string, data []byte, perm os.FileMode) {
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, perm)
	if err != nil {
		log.Fatalf("opening %s: %v", path, err)
	}
	defer file.Close()

	if err := pem.Encode(file, &pem.Block{Type: blockType, Bytes: data}); err != nil {
		log.Fatalf("writing %s: %v", path, err)
	}
}
