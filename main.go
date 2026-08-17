package main

import (
	"log"
	"net"
	"net/http"
)

const (
	httpAddr  = ":8080" // plain HTTP, only used to redirect
	httpsAddr = ":8443" // the real server
)

func main() {
	// A ServeMux is a router: it maps URL paths to handlers.
	// Previously we used http.Handle(...), which writes into a hidden
	// package-level mux (http.DefaultServeMux). Making our own is clearer
	// and avoids surprises when a library registers routes behind our back.
	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.Dir("./static")))

	// Start the HTTP redirect server in a goroutine so it runs alongside
	// the HTTPS one. `go f()` means "run f concurrently and keep going".
	go func() {
		log.Printf("HTTP  listening on %s (redirects to HTTPS)", httpAddr)
		err := http.ListenAndServe(httpAddr, http.HandlerFunc(redirectToHTTPS))
		// This only returns when the server stops, which is always an error.
		log.Fatal("http server:", err)
	}()

	log.Printf("HTTPS listening on %s", httpsAddr)
	log.Print("open https://10.252.176.230:8443 on the laptop")

	err := http.ListenAndServeTLS(httpsAddr, "certs/cert.pem", "certs/key.pem", mux)
	log.Fatal("https server:", err)
}

// redirectToHTTPS sends any plain-HTTP visitor to the HTTPS port.
//
// A handler receives the request (r) and a writer (w) it uses to build the
// response. r.Host is whatever the client typed, e.g. "10.252.176.230:8080",
// so we strip the port and substitute our HTTPS one.
func redirectToHTTPS(w http.ResponseWriter, r *http.Request) {
	host, _, err := net.SplitHostPort(r.Host)
	if err != nil {
		host = r.Host // no port in the header; use it as-is
	}

	target := "https://" + net.JoinHostPort(host, "8443") + r.URL.RequestURI()

	// 308 = permanent redirect that preserves the HTTP method.
	http.Redirect(w, r, target, http.StatusPermanentRedirect)
}
