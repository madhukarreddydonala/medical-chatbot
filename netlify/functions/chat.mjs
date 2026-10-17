const json = (body, status) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });

export default async (request) => {
  if (request.method !== "POST") {
    return json({ error: "Use POST for chat requests." }, 405);
  }

  const backendUrl = process.env.BACKEND_URL;
  const proxyToken = process.env.NETLIFY_PROXY_TOKEN;
  if (!backendUrl || !proxyToken) {
    return json({ error: "The chat service is not configured yet." }, 503);
  }

  let backendOrigin;
  try {
    const parsedBackend = new URL(backendUrl);
    if (parsedBackend.protocol !== "https:" || parsedBackend.pathname !== "/") {
      return json({ error: "The chat service is not configured securely." }, 503);
    }
    backendOrigin = parsedBackend.origin;
  } catch {
    return json({ error: "The chat service is not configured securely." }, 503);
  }

  const visitorKey = request.headers.get("x-groq-api-key") || "";
  if (!visitorKey || visitorKey.length > 512) {
    return json({ error: "Enter a valid Groq API key to continue." }, 400);
  }

  try {
    const body = await request.text();
    if (Buffer.byteLength(body, "utf8") > 32 * 1024) {
      return json({ error: "The request is too large." }, 413);
    }

    const upstream = await fetch(`${backendOrigin}/api/chat`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "application/json",
        "x-groq-api-key": visitorKey,
        "x-netlify-proxy-token": proxyToken,
      },
      body,
      signal: AbortSignal.timeout(55_000),
    });
    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        "content-type": upstream.headers.get("content-type") || "application/json",
        "cache-control": "no-store",
      },
    });
  } catch {
    return json({ error: "The chat service could not be reached. Please try again." }, 502);
  }
};

export const config = {
  path: "/api/chat",
  rateLimit: {
    windowLimit: 8,
    windowSize: 60,
    aggregateBy: ["ip", "domain"],
  },
};
