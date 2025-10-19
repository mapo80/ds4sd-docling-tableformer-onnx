using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Encodings.Web;

namespace TableFormerTorchSharpSdk.Utilities;

internal static class JsonCanonicalizer
{
    public static string GetCanonicalJson(JsonElement element)
    {
        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions
        {
            Indented = false,
            Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        }))
        {
            WriteCanonicalValue(writer, element);
        }

        var raw = Encoding.UTF8.GetString(stream.ToArray());
        var asciiEscaped = EscapeNonAscii(raw);
        return NormalizeUnicodeEscapes(asciiEscaped);
    }

    private static void WriteCanonicalValue(Utf8JsonWriter writer, JsonElement element)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                writer.WriteStartObject();
                foreach (var property in element.EnumerateObject().OrderBy(p => p.Name, StringComparer.Ordinal))
                {
                    writer.WritePropertyName(property.Name);
                    WriteCanonicalValue(writer, property.Value);
                }

                writer.WriteEndObject();
                break;
            case JsonValueKind.Array:
                writer.WriteStartArray();
                foreach (var item in element.EnumerateArray())
                {
                    WriteCanonicalValue(writer, item);
                }

                writer.WriteEndArray();
                break;
            case JsonValueKind.String:
                writer.WriteStringValue(element.GetString());
                break;
            case JsonValueKind.Number:
                writer.WriteRawValue(element.GetRawText());
                break;
            case JsonValueKind.True:
                writer.WriteBooleanValue(true);
                break;
            case JsonValueKind.False:
                writer.WriteBooleanValue(false);
                break;
            case JsonValueKind.Null:
                writer.WriteNullValue();
                break;
            case JsonValueKind.Undefined:
                writer.WriteNullValue();
                break;
            default:
                throw new NotSupportedException($"Unsupported JSON value kind: {element.ValueKind}");
        }
    }

    private static string EscapeNonAscii(string value)
    {
        var builder = new StringBuilder(value.Length);
        foreach (var ch in value)
        {
            if (ch > 0x7F)
            {
                builder.Append("\\u");
                builder.Append(((int)ch).ToString("x4"));
            }
            else
            {
                builder.Append(ch);
            }
        }

        return builder.ToString();
    }

    private static string NormalizeUnicodeEscapes(string value)
    {
        var builder = new StringBuilder(value.Length);
        for (var i = 0; i < value.Length; i++)
        {
            var ch = value[i];
            if (ch == '\\' && i + 5 < value.Length && value[i + 1] == 'u')
            {
                builder.Append("\\u");
                for (var j = i + 2; j < i + 6; j++)
                {
                    builder.Append(char.ToLowerInvariant(value[j]));
                }

                i += 5;
            }
            else
            {
                builder.Append(ch);
            }
        }

        return builder.ToString();
    }
}
