-- LaTeX only: render inline code with \path (xurl) so long paths break at / _ - instead of overflowing
function Code(el)
  if FORMAT:match("latex") and not el.text:find("[{}%%]") then
    return pandoc.RawInline("latex", "\\path{" .. el.text .. "}")
  end
end
